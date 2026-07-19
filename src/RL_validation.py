"""
Industry Standard Baseline Evaluation

Runs the standard autoclave cycle through the 1D FD environment
and reports key metrics.

Expected: ~150 min cycle, full cure, safe thermal profile
"""

import numpy as np
import sys
sys.path.append('.')  # Adjust to your path

# Import from your environment file
from Rl_Gym_compo import CompositeCuringEnv, rollout_policy, get_industry_standard_action


def evaluate_industry_standard():
    """Run and analyze industry standard cycle."""

    print("\n" + "="*80)
    print("INDUSTRY STANDARD BASELINE EVALUATION")
    print("="*80)
    print("\nCycle Profile:")
    print("  0-2400s (40 min):  Ramp 2°C/min to 80°C")
    print("  2400-4200s (30 min): Hold 80°C")
    print("  4200-6600s (27 min): Ramp 3°C/min to 150°C")
    print("  6600-13200s (110 min): Hold 150°C")
    print("  >13200s: Cool at 2°C/min")

    # Create environment with KIT SCER1 params
    env = CompositeCuringEnv(
        thickness=0.015,
        nx=21,
        dt_step=10.0,
        max_time=14400,  # 4 hours
        H_reaction=500000.0,
        k0=8.5e4,
        Ea=67600.0,
        n_order=1.6,
    )

    print("\n" + "-"*80)
    print("Running simulation...")
    print("-"*80)

    # Run baseline policy
    history, total_reward, success = rollout_policy(
        env,
        get_industry_standard_action,
        max_steps=env.max_steps
    )

    # Extract arrays
    t = np.array(history['t'])
    T_surface = np.array(history['T_surface'])
    T_center = np.array(history['T_center'])
    T_oven = np.array(history['T_oven'])
    alpha_center = np.array(history['alpha_center'])

    # Compute metrics
    cycle_time = t[-1]
    cycle_time_min = cycle_time / 60

    T_center_max = np.max(T_center)
    T_center_max_C = T_center_max - 273

    T_surface_max = np.max(T_surface)
    T_surface_max_C = T_surface_max - 273

    thermal_lag_max = np.max(T_surface - T_center)
    thermal_lag_min = np.min(T_surface - T_center)

    alpha_final_center = alpha_center[-1]
    alpha_final_avg = np.mean(alpha_center[-1])  # Last timestep

    # Time to reach key cure milestones
    idx_90 = np.where(alpha_center >= 0.90)[0]
    idx_95 = np.where(alpha_center >= 0.95)[0]
    time_to_90 = t[idx_90[0]] / 60 if len(idx_90) > 0 else None
    time_to_95 = t[idx_95[0]] / 60 if len(idx_95) > 0 else None

    # Safety check
    T_limit = 453  # 180°C
    exceeded_limit = np.any(T_center > T_limit)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n{'TIMING':.<40} {cycle_time_min:>12.1f} min")
    print(f"{'  Time to 90% cure':.<40} {time_to_90:>12.1f} min" if time_to_90 else "N/A")
    print(f"{'  Time to 95% cure':.<40} {time_to_95:>12.1f} min" if time_to_95 else "N/A")

    print(f"\n{'TEMPERATURE PROFILE':.<40}")
    print(f"{'  T_oven (max)':.<40} {np.max(T_oven) - 273:>12.1f}°C")
    print(f"{'  T_surface (max)':.<40} {T_surface_max_C:>12.1f}°C")
    print(f"{'  T_center (max)':.<40} {T_center_max_C:>12.1f}°C")
    print(f"{'  Thermal Lag (peak)':.<40} {thermal_lag_max:>12.1f} K")
    print(f"{'  Thermal Lag (min)':.<40} {thermal_lag_min:>12.1f} K")

    print(f"\n{'CURE STATE':.<40}")
    print(f"{'  α_center (final)':.<40} {alpha_final_center:>12.3f}")
    print(f"{'  Success (α≥0.95)':.<40} {'YES ✅' if alpha_final_center >= 0.95 else 'NO ❌':>12}")

    print(f"\n{'SAFETY':.<40}")
    print(f"{'  T_center exceeds 180°C':.<40} {'YES ⚠️' if exceeded_limit else 'NO ✅':>12}")
    print(f"{'  Thermal gradient control':.<40} {'GOOD' if thermal_lag_max < 20 else 'MARGINAL' if thermal_lag_max < 30 else 'POOR':>12}")

    print(f"\n{'REWARD':.<40} {total_reward:>12.2f}")

    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)

    if success and not exceeded_limit:
        print("✅ Industry Standard Cycle: SAFE and EFFECTIVE")
        print("   Baseline is well-designed. RL can optimize from here.")
        margin = 180 - T_center_max_C
        print(f"   Safety margin: {margin:.1f}°C")
    elif success and exceeded_limit:
        print("⚠️  Industry Standard Cycle: MARGINAL (Exotherm Risk)")
        print("   Cycle achieves cure but thermal safety is compromised.")
        print("   RL MUST find a safer alternative.")
    else:
        print("❌ Industry Standard Cycle: FAILS")
        print("   Even the standard approach doesn't work on this thick part.")
        print("   RL has a critical problem to solve.")

    print("\n" + "="*80)

    return {
        'cycle_time_min': cycle_time_min,
        'T_center_max': T_center_max,
        'T_surface_max': T_surface_max,
        'thermal_lag_max': thermal_lag_max,
        'alpha_final': alpha_final_center,
        'success': success,
        'exceeded_limit': exceeded_limit,
        'reward': total_reward,
        'history': history,
        't': t,
        'T_center': T_center,
        'T_surface': T_surface,
        'T_oven': T_oven,
        'alpha_center': alpha_center,
    }


def plot_baseline(results):
    """Plot baseline cycle for visualization."""
    import matplotlib.pyplot as plt

    t = results['t'] / 60  # Convert to minutes
    T_center = results['T_center'] - 273
    T_surface = results['T_surface'] - 273
    T_oven = results['T_oven'] - 273
    alpha = results['alpha_center']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Temperature
    ax = axes[0, 0]
    ax.plot(t, T_oven, 'k--', linewidth=2, label='Oven Setpoint', alpha=0.7)
    ax.plot(t, T_surface, 'b-', linewidth=2, label='Surface (Tool)')
    ax.plot(t, T_center, 'r-', linewidth=2.5, label='Center (CRITICAL)')
    ax.axhline(y=180, color='red', linestyle=':', alpha=0.5, label='180°C Safety Limit')
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('Temperature [°C]', fontsize=11)
    ax.set_title('Industry Standard: Temperature Profile', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Degree of Cure
    ax = axes[0, 1]
    ax.plot(t, alpha, 'g-', linewidth=2.5)
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Cure Target')
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('Degree of Cure α', fontsize=11)
    ax.set_title('Cure Progress', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Thermal Lag
    ax = axes[1, 0]
    lag = T_surface - T_center
    ax.plot(t, lag, 'purple', linewidth=2)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10°C (Design Threshold)')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax.set_xlabel('Time [min]', fontsize=11)
    ax.set_ylabel('Lag: T_surface - T_center [K]', fontsize=11)
    ax.set_title('Thermal Gradient', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary Metrics
    ax = axes[1, 1]
    ax.axis('off')

    success = results['success']
    exceeded = results['exceeded_limit']
    cycle_time = results['cycle_time_min']
    T_max = results['T_center_max'] - 273
    lag_max = results['thermal_lag_max']

    summary_text = f"""
    BASELINE SUMMARY
    
    Cycle Time:        {cycle_time:.1f} min
    Final α:           {results['alpha_final']:.3f}
    Success:           {'YES ✅' if success else 'NO ❌'}
    
    T_center (max):    {T_max:.1f}°C
    Safety Status:     {'SAFE ✅' if not exceeded else 'AT RISK ⚠️'}
    
    Thermal Lag (max): {lag_max:.1f} K
    Quality Control:   {'GOOD ✅' if lag_max < 20 else 'FAIR' if lag_max < 30 else 'POOR ❌'}
    
    Next Step: RL Training
    Goal: Beat this baseline by 15-20%
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('baseline_industry_standard.png', dpi=150, bbox_inches='tight')
    print("Saved plot: baseline_industry_standard.png")
    plt.show()


if __name__ == "__main__":
    results = evaluate_industry_standard()
    plot_baseline(results)
