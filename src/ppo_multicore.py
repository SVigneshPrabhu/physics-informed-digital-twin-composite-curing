"""
PPO Training Script - Composite Curing Optimization (OPTIMIZED MULTIPROCESS - WINDOWS FIXED)

Key improvements:
  - SubprocVecEnv for true parallelization
  - Windows-compatible multiprocessing setup
  - Fixed seed() issue with Monitor wrapper
  - Configurable number of worker processes
  - CPU affinity for better cache locality
"""

import os
import sys
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from Rl_Gym_compo import CompositeCuringEnv, get_industry_standard_multizone, rollout_policy


# Directories
LOG_DIR = "logs_final/"
MODEL_DIR = "models_final/"
RESULTS_DIR = "results_final/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_optimal_num_envs():
    """Automatically determine optimal number of parallel environments."""
    num_cpus = os.cpu_count()
    # Use 70% of available CPUs, minimum 2
    num_envs = max(2, int(num_cpus * 0.7))
    print(f"System has {num_cpus} CPUs -> Using {num_envs} parallel environments")
    return num_envs


def make_env(rank):
    """
    Create a single environment factory function.
    This must be defined at module level for Windows multiprocessing.
    """
    def _init():
        env = CompositeCuringEnv(
            thickness=0.015,
            nx=21,
            dt_step=10.0,
            max_time=14400,
            H_reaction=500000.0,
            k0=8.5e4,
            Ea=67600.0,
            n_order=1.6,
            tau_tool=300.0,
        )

        # Create monitor directory
        worker_log_dir = os.path.join(LOG_DIR, f"worker_{rank}")
        os.makedirs(worker_log_dir, exist_ok=True)

        # Wrap with Monitor (before seeding to avoid seed() call on Monitor)
        env = Monitor(env, worker_log_dir)

        # Seed the underlying environment through reset
        env.reset(seed=rank)

        return env

    return _init


def train_ppo(total_timesteps=500000, num_envs=None):
    """Train PPO agent with multiprocessing."""

    if num_envs is None:
        num_envs = get_optimal_num_envs()

    print("\n" + "=" * 80)
    print("PPO TRAINING: COMPOSITE CURING OPTIMIZATION (MULTIPROCESS)")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {num_envs}")
    print(f"Timesteps per env: {total_timesteps / num_envs:,.0f}")
    print("=" * 80 + "\n")

    # Create parallel environments (TRUE parallelization)
    # Use lambda to ensure proper closure
    env_fns = [make_env(i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)

    # Optimized hyperparameters for parallel training
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per environment before update
        batch_size=128,  # Larger batch size for parallel efficiency
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        device="auto",
        seed=42,
    )

    print("Training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            tb_log_name="ppo_composite_multiprocess"
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    finally:
        # Save model
        model_path = os.path.join(MODEL_DIR, "ppo_final")
        model.save(model_path)
        print(f"\n✅ Model saved to {model_path}")

        # Close environments
        env.close()

    return model


def evaluate_model(model_path, num_episodes=5):
    """Evaluate trained model vs baseline."""

    print("\n" + "=" * 80)
    print("EVALUATION: AI vs INDUSTRY STANDARD")
    print("=" * 80 + "\n")

    model = PPO.load(model_path)

    env = CompositeCuringEnv(
        thickness=0.015,
        nx=21,
        dt_step=10.0,
        max_time=14400,
        H_reaction=500000.0,
        k0=8.5e4,
        Ea=67600.0,
        n_order=1.6,
        tau_tool=300.0,
    )

    # AI episodes
    ai_results = []
    print("Running AI episodes...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        history = env.get_episode_history()
        success = history['alpha_center'][-1] >= 0.95
        cycle_time = history['t'][-1] / 60
        T_max = np.max(history['T_center']) - 273
        alpha_final = history['alpha_center'][-1]

        ai_results.append({
            'cycle_time': cycle_time,
            'T_max': T_max,
            'alpha_final': alpha_final,
            'success': success,
            'history': history,
            'reward': total_reward
        })

        print(
            f"Episode {episode + 1}: Time={cycle_time:.1f}min, α={alpha_final:.3f}, Success={'✅' if success else '❌'}")

    # Baseline episodes
    baseline_results = []
    print("\nRunning Baseline episodes...\n")

    for episode in range(num_episodes):
        hist, reward, success = rollout_policy(
            env,
            get_industry_standard_multizone,
            max_steps=env.max_steps
        )

        cycle_time = hist['t'][-1] / 60
        T_max = np.max(hist['T_center']) - 273
        alpha_final = hist['alpha_center'][-1]

        baseline_results.append({
            'cycle_time': cycle_time,
            'T_max': T_max,
            'alpha_final': alpha_final,
            'success': success,
            'history': hist,
            'reward': reward
        })

        print(
            f"Episode {episode + 1}: Time={cycle_time:.1f}min, α={alpha_final:.3f}, Success={'✅' if success else '❌'}")

    # Compare
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    ai_times = [r['cycle_time'] for r in ai_results]
    baseline_times = [r['cycle_time'] for r in baseline_results]

    ai_success = sum(1 for r in ai_results if r['success'])
    baseline_success = sum(1 for r in baseline_results if r['success'])

    ai_avg_time = np.mean(ai_times)
    baseline_avg_time = np.mean(baseline_times)
    improvement = (baseline_avg_time - ai_avg_time) / baseline_avg_time * 100

    print(f"\nAI Performance:")
    print(f"  Avg cycle time: {ai_avg_time:.1f} min")
    print(f"  Success rate: {ai_success}/{num_episodes}")

    print(f"\nBaseline Performance:")
    print(f"  Avg cycle time: {baseline_avg_time:.1f} min")
    print(f"  Success rate: {baseline_success}/{num_episodes}")

    print(f"\n🎯 Improvement: {improvement:+.1f}%")

    if improvement > 0:
        print(f"✅ AI BEATS BASELINE")
    else:
        print(f"⚠️  Baseline still faster")

    print("=" * 80)

    env.close()
    return ai_results, baseline_results


def plot_results(ai_results, baseline_results):
    """Generate comparison plots."""

    ai_hist = ai_results[0]['history']
    baseline_hist = baseline_results[0]['history']

    t_ai = np.array(ai_hist['t']) / 60
    t_baseline = np.array(baseline_hist['t']) / 60

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature
    ax = axes[0, 0]
    ax.plot(t_ai, np.array(ai_hist['T_center']) - 273, 'r-', linewidth=2.5, label='AI')
    ax.plot(t_baseline, np.array(baseline_hist['T_center']) - 273, 'k-', linewidth=2, alpha=0.6, label='Baseline')
    ax.axhline(y=180, color='red', linestyle=':', alpha=0.7)
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('T_center [°C]', fontsize=11)
    ax.set_title('Temperature: AI vs Baseline', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cure
    ax = axes[0, 1]
    ax.plot(t_ai, ai_hist['alpha_center'], 'r-', linewidth=2.5, label='AI')
    ax.plot(t_baseline, baseline_hist['alpha_center'], 'k-', linewidth=2, alpha=0.6, label='Baseline')
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('Degree of Cure α', fontsize=11)
    ax.set_title('Cure Progress: AI vs Baseline', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tool lag
    ax = axes[1, 0]
    ax.plot(t_ai, np.array(ai_hist['T_air_tool']) - 273, 'b--', linewidth=1, alpha=0.7, label='AI Oven Air')
    ax.plot(t_ai, np.array(ai_hist['T_tool_plate']) - 273, 'b-', linewidth=2, label='AI Tool Plate')
    ax.plot(t_baseline, np.array(baseline_hist['T_air_tool']) - 273, 'k--', linewidth=1, alpha=0.5,
            label='Baseline Oven Air')
    ax.plot(t_baseline, np.array(baseline_hist['T_tool_plate']) - 273, 'k-', linewidth=1.5, alpha=0.5,
            label='Baseline Tool Plate')
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.set_title('Tool Thermal Lag', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1, 1]
    ax.axis('off')

    ai_cycle = ai_results[0]['cycle_time']
    baseline_cycle = baseline_results[0]['cycle_time']
    improvement_pct = (baseline_cycle - ai_cycle) / baseline_cycle * 100

    summary = f"""
    FINAL RESULTS

    AI Policy:
      Cycle Time:  {ai_cycle:.1f} min
      T_max:       {ai_results[0]['T_max']:.1f}°C
      Final α:     {ai_results[0]['alpha_final']:.3f}
      Success:     {'YES ✅' if ai_results[0]['success'] else 'NO ❌'}

    Baseline:
      Cycle Time:  {baseline_cycle:.1f} min
      T_max:       {baseline_results[0]['T_max']:.1f}°C
      Final α:     {baseline_results[0]['alpha_final']:.3f}
      Success:     {'YES ✅' if baseline_results[0]['success'] else 'NO ❌'}

    IMPROVEMENT: {improvement_pct:+.1f}%

    Features:
      ✓ Tool thermal mass (tau=300s)
      ✓ Multizone control
      ✓ KIT SCER1 kinetics
      ✓ Potential-based shaping
      ✓ MULTIPROCESS (SubprocVecEnv)
    """

    ax.text(0.05, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'final_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Windows compatibility: ensure proper multiprocessing context
    if sys.platform == "win32":
        # Use spawn method for Windows (required for multiprocessing)
        mp.set_start_method('spawn', force=True)

    # Train with automatic process detection
    num_envs = get_optimal_num_envs()
    model = train_ppo(total_timesteps=500000, num_envs=num_envs)

    # Evaluate
    model_path = os.path.join(MODEL_DIR, "ppo_final")
    ai_results, baseline_results = evaluate_model(model_path, num_episodes=5)

    # Plot
    plot_results(ai_results, baseline_results)

    print("\n✅ COMPLETE!")
