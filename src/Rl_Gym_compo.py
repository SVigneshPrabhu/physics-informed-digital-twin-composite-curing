""""
Composite Curing Environment - Gymnasium Wrapper (FIXED CONSTRAINT INTEGRATED)

Features:
  1. Physics: KIT SCER1 (High Ea) + Tool Lag (tau=300s)
  2. Control: Multizone (Tool/Bag)
  3. Logic: Near-Miss Constraint Penalties + Potential-Based Reward Shaping
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
from collections import deque


# =============================================================================
# CONSTRAINT MANAGER (Helper Class)
# =============================================================================

class NearMissConstraintManager:
    """Manages manufacturing constraints based on historical failure data."""

    def __init__(self):
        self.constraints = [
            # 1. Ramp Rate Constraint: Must be > 0.6°C/min at t=60-65 min
            {'type': 'min', 'metric': 'ramp', 'val': 0.01, 't_start': 3600, 't_end': 3900, 'severity': 1.0},
            # 2. Thermal Lag Constraint: Max 20K lag at t=120-135 min
            {'type': 'max', 'metric': 'lag', 'val': 20.0, 't_start': 7200, 't_end': 8100, 'severity': 0.8},
        ]
        self.violation_log = []

    def evaluate(self, current_time, ramp_rate, thermal_lag, penalty_scale):
        penalty = 0.0

        for c in self.constraints:
            if c['t_start'] <= current_time <= c['t_end']:
                violation = 0.0

                # RAMP CHECK
                if c['metric'] == 'ramp':
                    if ramp_rate < c['val']:  # Too slow
                        violation = (c['val'] - ramp_rate)

                # LAG CHECK
                elif c['metric'] == 'lag':
                    if thermal_lag > c['val']:  # Too high
                        violation = (thermal_lag - c['val'])

                # Apply Penalty
                if violation > 0:
                    p = -1.0 * violation * c['severity'] * penalty_scale
                    penalty += p

        return penalty

    def reset(self):
        self.violation_log = []


# =============================================================================
# MAIN ENVIRONMENT
# =============================================================================

class CompositeCuringEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 thickness=0.015,
                 nx=21,
                 dt_step=10.0,
                 max_time=14400,
                 T_initial=298.0,
                 # Physics
                 rho=1580, Cp=1200, k_ref=0.28, H_reaction=500000.0,
                 k0=8.5e4, Ea=67600.0, n_order=1.6, R=8.314,
                 tau_tool=300.0,
                 # Constraints
                 use_near_miss_constraints=False,
                 penalty_scale=10.0,
                 render_mode=None):

        self.thickness = thickness
        self.nx = nx
        self.dx = thickness / (nx - 1)
        self.dt_step = dt_step
        self.max_time = max_time
        self.max_steps = int(max_time / dt_step)
        self.T_initial = T_initial

        # Physics Params
        self.rho = rho;
        self.Cp = Cp;
        self.k_ref = k_ref
        self.H_reaction = H_reaction;
        self.k0 = k0;
        self.Ea = Ea
        self.n_order = n_order;
        self.R = R
        self.tau_tool = tau_tool

        # Grid Definition (FIXED: Added missing i_center)
        self.x = np.linspace(0, thickness, nx)
        self.i_center = nx // 2

        # Constraint Logic
        self.use_near_miss_constraints = use_near_miss_constraints
        self.penalty_scale = penalty_scale
        self.constraint_manager = NearMissConstraintManager()
        self.T_center_history = deque(maxlen=6)  # For smoothing ramp calc

        # State
        self.T_tool_plate = T_initial
        self.T = None
        self.alpha = None

        # Bounds
        self.T_air_min = 298.0
        self.T_air_max = 453.0
        self.max_ramp_rate = 3.0 / 60.0

        # Actions: [-1, 1] Normalized
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Obs: 7D Normalized
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Reward Trackers
        self.alpha_prev = 0.0
        self._passed_40C = False
        self._passed_60C = False
        self._passed_100C = False
        self._passed_130C = False

        self.episode_history = {}

    def _k_thermal(self, T, alpha):
        return np.maximum(self.k_ref * (1 - alpha * 0.2), 0.01)

    def _kinetic_rate(self, T, alpha):
        if T <= 0: return 0.0
        exponent = np.clip(-self.Ea / (self.R * T), -700, 0)
        return self.k0 * np.exp(exponent) * np.maximum(1 - alpha, 0) ** self.n_order

    def _dynamics(self, t, y):
        T_vec = y[:self.nx]
        alpha_vec = np.clip(y[self.nx:], 0.0, 1.0)
        dT_dt = np.zeros(self.nx)
        dalpha_dt = np.zeros(self.nx)

        for i in range(1, self.nx - 1):
            T_i = T_vec[i];
            alpha_i = alpha_vec[i]
            k_left = 0.5 * (self._k_thermal(T_vec[i - 1], alpha_vec[i - 1]) + self._k_thermal(T_i, alpha_i))
            k_right = 0.5 * (self._k_thermal(T_i, alpha_i) + self._k_thermal(T_vec[i + 1], alpha_vec[i + 1]))
            d2T_dx2 = (k_right * (T_vec[i + 1] - T_i) - k_left * (T_i - T_vec[i - 1])) / (self.dx ** 2)
            dalpha_i = self._kinetic_rate(T_i, alpha_i)
            dT_dt[i] = (d2T_dx2 + self.H_reaction * dalpha_i) / (self.rho * self.Cp)
            dalpha_dt[i] = dalpha_i

        dT_dt[0] = 0;
        dT_dt[self.nx - 1] = 0
        dalpha_dt[0] = self._kinetic_rate(T_vec[0], alpha_vec[0])
        dalpha_dt[self.nx - 1] = self._kinetic_rate(T_vec[self.nx - 1], alpha_vec[self.nx - 1])
        return np.concatenate([dT_dt, dalpha_dt])

    def _integrate_step(self, action):
        # Scale Action [-1, 1] -> [298, 453]
        T_tool_K = self.T_air_min + (action[0] + 1.0) * 0.5 * (self.T_air_max - self.T_air_min)
        T_bag_K = self.T_air_min + (action[1] + 1.0) * 0.5 * (self.T_air_max - self.T_air_min)

        # Ramp Limit
        self.T_air_tool_side = np.clip(T_tool_K, self.T_air_tool_side - self.max_ramp_rate * self.dt_step,
                                       self.T_air_tool_side + self.max_ramp_rate * self.dt_step)
        self.T_air_bag_side = np.clip(T_bag_K, self.T_air_bag_side - self.max_ramp_rate * self.dt_step,
                                      self.T_air_bag_side + self.max_ramp_rate * self.dt_step)

        # Tool Lag
        self.T_tool_plate += (self.dt_step / self.tau_tool) * (self.T_air_tool_side - self.T_tool_plate)

        # Solve
        y0 = np.concatenate([self.T, self.alpha])

        def rhs(t, y):
            y[0] = self.T_tool_plate;
            y[self.nx - 1] = self.T_air_bag_side
            return self._dynamics(t, y)

        sol = solve_ivp(rhs, [0, self.dt_step], y0, method='RK45', max_step=1.0)
        self.T = sol.y[:self.nx, -1]
        self.alpha = np.clip(sol.y[self.nx:, -1], 0.0, 1.0)

    def _get_obs(self):
        # 7D Normalized Observation
        t_min, t_max = 298.0, 453.0
        T_surf = self.T[0];
        T_cent = self.T[self.i_center]
        dalpha = self._kinetic_rate(T_cent, self.alpha[self.i_center])

        return np.array([
            (T_surf - t_min) / (t_max - t_min),
            (T_cent - t_min) / (t_max - t_min),
            (self.T_tool_plate - t_min) / (t_max - t_min),
            np.clip((T_surf - T_cent) / 50.0, -1, 1),
            self.alpha[self.i_center],
            np.clip(dalpha / 0.002, 0, 1),
            self.current_step / self.max_steps
        ], dtype=np.float32)

    def _calculate_reward(self):
        T_cent = self.T[self.i_center]
        alpha = self.alpha[self.i_center]

        # 1. HARD FAILURE
        if T_cent > 453.0: return -100.0, True

        # 2. PROGRESS (Strongest Driver)
        r_prog = 500.0 * (alpha - self.alpha_prev)
        self.alpha_prev = alpha

        # 3. WARMUP (Bribe to heat up)
        r_warm = 0.0
        if alpha < 0.2: r_warm = 0.02 * (T_cent - 298.0)

        # 4. MILESTONES
        r_mil = 0.0
        if T_cent > 313 and not self._passed_40C: r_mil += 10; self._passed_40C = True
        if T_cent > 333 and not self._passed_60C: r_mil += 20; self._passed_60C = True
        if T_cent > 373 and not self._passed_100C: r_mil += 30; self._passed_100C = True
        if T_cent > 403 and not self._passed_130C: r_mil += 50; self._passed_130C = True

        # 5. CONSTRAINT PENALTIES (The "Near Miss" Logic)
        r_cons = 0.0
        if self.use_near_miss_constraints:
            # Calculate smooth ramp rate
            self.T_center_history.append(T_cent)
            if len(self.T_center_history) >= 2:
                dT = self.T_center_history[-1] - self.T_center_history[0]
                dt = (len(self.T_center_history) - 1) * self.dt_step
                ramp_K_s = dT / dt
            else:
                ramp_K_s = 0.0

            lag_K = abs(self.T[0] - T_cent)

            # Evaluate using Manager
            r_cons = self.constraint_manager.evaluate(
                self.current_time, ramp_K_s, lag_K, self.penalty_scale
            )

        # 6. Success
        done = False
        if alpha >= 0.95:
            r_prog += 500.0
            done = True

        return r_prog + r_warm + r_mil + r_cons - 0.01, done  # Tiny time penalty -0.01

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.T = np.full(self.nx, self.T_initial, dtype=np.float32)
        self.alpha = np.zeros(self.nx, dtype=np.float32)
        self.T_tool_plate = self.T_initial
        self.T_air_tool_side = self.T_initial
        self.T_air_bag_side = self.T_initial

        self.current_time = 0;
        self.current_step = 0;
        self.alpha_prev = 0.0
        self._passed_40C = False;
        self._passed_60C = False
        self._passed_100C = False;
        self._passed_130C = False

        self.T_center_history.clear()
        self.constraint_manager.reset()

        self.episode_history = {'t': [], 'T_surface': [], 'T_center': [], 'T_tool_plate': [], 'T_air_tool': [],
                                'alpha_center': [], 'reward': []}
        return self._get_obs(), {}

    def step(self, action):
        self._integrate_step(action)
        self.current_time += self.dt_step
        self.current_step += 1

        obs = self._get_obs()
        reward, terminated = self._calculate_reward()
        truncated = self.current_step >= self.max_steps

        # History
        self.episode_history['t'].append(self.current_time)
        self.episode_history['T_surface'].append(float(self.T[0]))
        self.episode_history['T_center'].append(float(self.T[self.i_center]))
        self.episode_history['T_tool_plate'].append(float(self.T_tool_plate))
        self.episode_history['T_air_tool'].append(float(self.T_air_tool_side))
        self.episode_history['alpha_center'].append(float(self.alpha[self.i_center]))
        self.episode_history['reward'].append(reward)

        return obs, reward, terminated, truncated, {}

    def get_episode_history(self):
        return {k: np.array(v) for k, v in self.episode_history.items()}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_industry_standard_multizone(t_elapsed):
    """
    Industry standard cycle adapted for multizone control.
    Returns normalized action [-1, 1].
    """

    def standard_cycle(t):
        if t < 2400:
            return 298.0 + (t / 60) * 2
        elif t < 4200:
            return 353.0
        elif t < 6600:
            return 353.0 + ((t - 4200) / 60) * 3
        elif t < 13200:
            return 423.0
        else:
            return 423.0 - ((t - 13200) / 60) * 2

    T_tool = standard_cycle(t_elapsed)
    t_bag_delayed = max(0, t_elapsed - 300)
    T_bag = standard_cycle(t_bag_delayed)

    # Normalize to [-1, 1] for environment
    def normalize(val):
        return (val - 298.0) / (453.0 - 298.0) * 2.0 - 1.0

    return np.array([normalize(T_tool), normalize(T_bag)], dtype=np.float32)


def rollout_policy(env, policy_func, max_steps=None):
    """Execute episode using a policy function."""
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    obs, _ = actual_env.reset()

    total_reward = 0.0
    limit = max_steps or actual_env.max_steps

    for step in range(limit):
        action = policy_func(actual_env.current_time)
        obs, reward, terminated, truncated, info = actual_env.step(action)
        total_reward += reward
        if terminated or truncated: break

    history = actual_env.get_episode_history()
    success = False
    if 'alpha_center' in history and len(history['alpha_center']) > 0:
        success = history['alpha_center'][-1] >= 0.95

    return history, total_reward, success


if __name__ == "__main__":
    env = CompositeCuringEnv()
    print("Environment Loaded Correctly with Helper Functions.")
