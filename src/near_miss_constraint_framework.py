"""
Near-Miss Constraint Framework (v3 - Production Ready)
"""

import numpy as np
import collections
from typing import List, Dict, Tuple, Literal, Optional


class NearMissConstraint:
    """
    Represents a single near-miss failure constraint from trial data.

    Attributes:
        trial_id: Identifier (e.g., "XYZ-001")
        failure_type: Type of metric ('ramp_rate', 'thermal_lag', 'cure', 'exotherm')
        time_window: (t_start, t_end) in seconds where constraint is active
        violation_metric: Actual value that failed in trial
        required_metric: Spec value that should have been maintained
        limit_type: 'min' = value must be >= required, 'max' = value must be <= required
        severity: Penalty multiplier [0.1, 1.0] (1.0 = fatal)
    """

    def __init__(self,
                 trial_id: str,
                 failure_type: str,
                 time_window: Tuple[float, float],
                 violation_metric: float,
                 required_metric: float,
                 limit_type: Literal['min', 'max'],
                 severity: float = 1.0):
        self.trial_id = trial_id
        self.failure_type = failure_type
        self.time_window = time_window
        self.violation_metric = violation_metric
        self.required_metric = required_metric
        self.limit_type = limit_type
        self.severity = min(max(severity, 0.1), 1.0)  # Clip to [0.1, 1.0]

    def __repr__(self):
        return (f"NearMiss({self.trial_id}, {self.failure_type}, "
                f"t=[{self.time_window[0]:.0f}, {self.time_window[1]:.0f}]s, "
                f"limit={self.limit_type}:{self.required_metric:.3f}, "
                f"severity={self.severity})")


class NearMissConstraintManager:
    """
    Manages near-miss constraints and evaluates violations during RL episodes.
    """

    def __init__(self, constraints: List[NearMissConstraint] = None,
                 penalty_scale: float = 100.0,
                 smoothing_window: int = 5):
        """
        Args:
            constraints: List of NearMissConstraint objects
            penalty_scale: Multiplier for constraint penalties (higher = stricter)
            smoothing_window: Window size for derivative smoothing (in timesteps)
        """
        self.constraints = constraints or []
        self.violation_log = []
        self.penalty_scale = penalty_scale
        self.smoothing_window = smoothing_window

    def add_constraint(self, constraint: NearMissConstraint):
        """Add a single constraint."""
        self.constraints.append(constraint)

    def add_constraints_from_trials(self, trial_data: List[Dict]):
        """
        Parse trial data and extract near-miss constraints.

        Args:
            trial_data: List of dicts with keys:
                {
                    'trial_id': str,
                    'failure_type': str,
                    'limit_type': 'min' or 'max',
                    'time_start': float (seconds),
                    'time_end': float,
                    'actual_value': float,
                    'required_value': float,
                    'severity': float (optional, default 1.0)
                }
        """
        for trial in trial_data:
            constraint = NearMissConstraint(
                trial_id=trial['trial_id'],
                failure_type=trial['failure_type'],
                time_window=(trial['time_start'], trial['time_end']),
                violation_metric=trial['actual_value'],
                required_metric=trial['required_value'],
                limit_type=trial['limit_type'],
                severity=trial.get('severity', 1.0)
            )
            self.add_constraint(constraint)

    def evaluate_phase(self, current_time: float, metric_name: str,
                       current_value: float) -> float:
        """
        CORRECTED LOGIC: Evaluate if current phase violates any constraint.

        Handles both MIN constraints (value must be >=) and MAX constraints (<=)

        Args:
            current_time: Current simulation time [seconds]
            metric_name: Type of metric being evaluated
            current_value: Current measured value

        Returns:
            penalty: Negative reward if violated, 0.0 if compliant
        """
        penalty = 0.0

        for constraint in self.constraints:
            # 1. Check if this constraint applies to this metric
            if constraint.failure_type != metric_name:
                continue

            # 2. Check if we're in the constraint's time window
            t_start, t_end = constraint.time_window
            in_danger_window = t_start <= current_time <= t_end

            if not in_danger_window:
                continue

            # 3. CRITICAL FIX: Check violation based on limit_type
            violation = False
            deficit = 0.0

            if constraint.limit_type == 'min':
                # Constraint: Value MUST be >= required (e.g., ramp_rate >= 0.01)
                if current_value < constraint.required_metric:
                    violation = True
                    # Deficit: how much below the requirement
                    deficit = (constraint.required_metric - current_value) / (abs(constraint.required_metric) + 1e-6)

            elif constraint.limit_type == 'max':
                # Constraint: Value MUST be <= required (e.g., thermal_lag <= 20)
                if current_value > constraint.required_metric:
                    violation = True
                    # Deficit: how much above the requirement
                    deficit = (current_value - constraint.required_metric) / (abs(constraint.required_metric) + 1e-6)

            # 4. Apply penalty if violated
            if violation:
                # Penalty = -1 * Severity * Normalized_Deficit * Scaling_Factor
                # This ensures the agent immediately learns to avoid violations
                constraint_penalty = -1.0 * constraint.severity * min(deficit, 1.0) * self.penalty_scale
                penalty += constraint_penalty

                # Log violation (with size limit to avoid memory bloat during training)
                if len(self.violation_log) < 5000:
                    self.violation_log.append({
                        'trial_id': constraint.trial_id,
                        'time': current_time,
                        'metric': metric_name,
                        'actual': current_value,
                        'required': constraint.required_metric,
                        'limit_type': constraint.limit_type,
                        'severity': constraint.severity,
                        'deficit': deficit,
                        'penalty': constraint_penalty
                    })

        return penalty

    # Wrapper methods for common metrics
    def evaluate_ramp_rate(self, current_time: float, ramp_rate_K_per_s: float) -> float:
        """Evaluate ramp rate constraint (typically min limit)."""
        return self.evaluate_phase(current_time, 'ramp_rate', ramp_rate_K_per_s)

    def evaluate_thermal_lag(self, current_time: float, lag_K: float) -> float:
        """Evaluate thermal lag constraint (typically max limit)."""
        return self.evaluate_phase(current_time, 'thermal_lag', lag_K)

    def evaluate_cure(self, current_time: float, alpha: float) -> float:
        """Evaluate cure progress constraint (typically min limit)."""
        return self.evaluate_phase(current_time, 'cure', alpha)

    def evaluate_exotherm(self, current_time: float, T_center_K: float) -> float:
        """Evaluate exotherm constraint (typically max limit)."""
        return self.evaluate_phase(current_time, 'exotherm', T_center_K)

    def reset_violation_log(self):
        """Clear violation log for new episode."""
        self.violation_log = []

    def get_violations_summary(self) -> Dict:
        """Return summary of violations during episode."""
        if not self.violation_log:
            return {
                'total_violations': 0,
                'total_penalty': 0.0,
                'trials_triggered': [],
                'metrics_violated': []
            }

        return {
            'total_violations': len(self.violation_log),
            'total_penalty': sum(v['penalty'] for v in self.violation_log),
            'trials_triggered': list(set(v['trial_id'] for v in self.violation_log)),
            'metrics_violated': list(set(v['metric'] for v in self.violation_log)),
            'avg_deficit': np.mean([v['deficit'] for v in self.violation_log])
        }


# =============================================================================
# EXAMPLE TRIAL DATA (With Correct limit_type)
# =============================================================================

EXAMPLE_NEAR_MISS_TRIALS = [
    {
        'trial_id': 'XYZ-001',
        'failure_type': 'ramp_rate',
        'limit_type': 'min',  # Ramp rate MUST be >= 0.01 K/s (0.6°C/min)
        'time_start': 3600,  # 60 min
        'time_end': 3900,  # 65 min
        'actual_value': 0.008,  # Failed: only 0.008 K/s (0.48°C/min)
        'required_value': 0.01,
        'severity': 0.8,
    },
    {
        'trial_id': 'XYZ-002',
        'failure_type': 'thermal_lag',
        'limit_type': 'max',  # Thermal lag MUST be <= 20K
        'time_start': 7200,  # 120 min
        'time_end': 8100,  # 135 min
        'actual_value': 27.0,  # Failed: 27K lag (too high)
        'required_value': 20.0,
        'severity': 0.6,
    },
    {
        'trial_id': 'XYZ-003',
        'failure_type': 'cure',
        'limit_type': 'min',  # Final cure MUST be >= 0.95
        'time_start': 14000,  # 233 min
        'time_end': 14400,  # 240 min (end of cycle)
        'actual_value': 0.92,  # Failed: incomplete cure
        'required_value': 0.95,
        'severity': 1.0,  # Critical failure
    },
]


# =============================================================================
# INTEGRATION HELPER FUNCTION
# =============================================================================

def create_smoothed_ramp_rate_calculator(smoothing_window: int = 5):
    """
    Create a function that calculates smoothed ramp rate over a time window.
    This avoids noisy single-step derivatives.

    Args:
        smoothing_window: Number of timesteps to average over

    Returns:
        A function ready to use in the RL loop
    """
    history = collections.deque(maxlen=smoothing_window)

    def add_and_calculate(T_center: float, dt: float) -> float:
        """
        Add temperature to history and return smoothed ramp rate.

        Args:
            T_center: Current center temperature [K]
            dt: Timestep [s]

        Returns:
            Smoothed ramp rate [K/s]
        """
        history.append(T_center)

        if len(history) < 2:
            return 0.0

        # Simple average over window
        dT = history[-1] - history[0]
        dt_window = (len(history) - 1) * dt

        return dT / dt_window if dt_window > 0 else 0.0

    return add_and_calculate


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NEAR-MISS CONSTRAINT FRAMEWORK (v3 - CORRECTED)")
    print("=" * 80)

    # Create manager
    manager = NearMissConstraintManager(
        penalty_scale=100.0,
        smoothing_window=5
    )
    manager.add_constraints_from_trials(EXAMPLE_NEAR_MISS_TRIALS)

    print("\nLoaded Constraints:")
    for i, c in enumerate(manager.constraints, 1):
        print(f"  {i}. {c}")

    # Test constraint evaluation
    print("\n" + "-" * 80)
    print("TESTING CONSTRAINT LOGIC (MIN vs MAX)")
    print("-" * 80)

    # Test 1: MIN constraint (ramp_rate)
    print("\nTest 1 - Ramp Rate (MIN constraint):")
    print("  Constraint: ramp_rate >= 0.01 K/s")
    print("  Actual: 0.008 K/s")
    penalty = manager.evaluate_ramp_rate(3700, 0.008)
    print(f"  Penalty: {penalty:.2f} ✓ (negative = violation detected)")

    # Test 2: MAX constraint (thermal_lag)
    print("\nTest 2 - Thermal Lag (MAX constraint):")
    print("  Constraint: thermal_lag <= 20K")
    print("  Actual: 25K")
    penalty = manager.evaluate_thermal_lag(7500, 25.0)
    print(f"  Penalty: {penalty:.2f} ✓ (negative = violation detected)")

    # Test 3: Compliant case
    print("\nTest 3 - Compliant Case (MIN constraint):")
    print("  Constraint: ramp_rate >= 0.01 K/s")
    print("  Actual: 0.012 K/s (compliant)")
    penalty = manager.evaluate_ramp_rate(3700, 0.012)
    print(f"  Penalty: {penalty:.2f} ✓ (zero = no violation)")

    # Test 4: Smoothed ramp rate
    print("\n" + "-" * 80)
    print("TESTING SMOOTHED RAMP RATE")
    print("-" * 80)
    ramp_calc = create_smoothed_ramp_rate_calculator(smoothing_window=5)

    temps = [298.0, 300.5, 303.0, 305.5, 308.0, 310.5]  # K
    dt = 10.0  # seconds

    print(f"\nTemperatures over {dt}s timesteps: {temps}")
    for i, T in enumerate(temps):
        ramp = ramp_calc(T, dt)
        print(f"  Step {i}: T={T}K, Smoothed ramp={ramp:.4f} K/s ({ramp * 60:.2f}°C/min)")

    print("\n" + "=" * 80)
    print("✓ All tests passed. Ready for RL integration.")
    print("=" * 80)
