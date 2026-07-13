"""
Perturbation-robustness testing across all three impairment combos.
Applies a brief external force to the torso mid-walk via
sim.data.xfrc_applied, then measures whether the model recovers or
falls -- giving us recovery rate, max tolerated force, and recovery
time (Miles's three requested perturbation metrics).

IMPORTANT AXIS NOTE: this model walks along the Y-AXIS (confirmed via
direct trajectory tracing early in the project), not the X-axis. So:
  - push_axis=1 (force in y) = SAGITTAL (forward/backward, along travel)
  - push_axis=0 (force in x) = LATERAL (sideways, perpendicular to travel)
An earlier version of this script had these swapped -- fixed here.

Two categories of failure are distinguished:
  - "invalid" trial: model fell BEFORE the push finished being applied
    (i.e. it fell on its own, unrelated to the perturbation) -- these
    are discarded from analysis, not counted as perturbation failures.
  - "real" trial: model survived through the push, then either
    recovered or fell as a genuine consequence of the push.
"""
from myosuite.utils import gym
import deprl
import numpy as np

TORSO_BODY_ID = 2          # confirmed: 0=world,1=root,2=torso,3=pelvis...
SAGITTAL_AXIS = 1           # y -- direction of travel, confirmed Day 1
LATERAL_AXIS = 0            # x -- perpendicular to travel

PUSH_DURATION_STEPS = 10    # push held for 10 steps (~0.1s at dt=0.01)
WARMUP_STEPS = 100          # let the model reach steady walking before pushing
RECOVERY_WINDOW = 200       # steps after push-end to judge recovery
MAX_STEPS = 1000


def apply_healthy_or_als(sim, strength_factor):
    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    scaled[:, 2] = original[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled


def apply_stroke(sim, paretic_strength, paretic_side="left"):
    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    idx = slice(40, 80) if paretic_side == "left" else slice(0, 40)
    scaled[idx, 2] = original[idx, 2] * paretic_strength
    sim.model.actuator_gainprm[:] = scaled


def get_body_weight_newtons(sim):
    total_mass = np.sum(sim.model.body_mass)
    return total_mass * 9.81


def run_perturbation_trial(policy, setup_fn, push_magnitude, push_axis,
                             jitter_std=None, seed=None, max_steps=MAX_STEPS):
    """
    Returns a dict with:
      status: "invalid" (fell before push completed -- discard),
              "recovered" (survived past the recovery window),
              "fell" (genuine perturbation-caused fall)
      fell_at_step, steps_after_push: only meaningful when status != "invalid"
    """
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    if setup_fn is not None:
        setup_fn(sim)

    obs, _ = env.reset()

    push_end_step = WARMUP_STEPS + PUSH_DURATION_STEPS
    terminated_step = None
    current_action = None
    steps_since_update = 0
    next_hold_duration = 1

    for step in range(max_steps):
        if WARMUP_STEPS <= step < push_end_step:
            sim.data.xfrc_applied[TORSO_BODY_ID, push_axis] = push_magnitude
        else:
            sim.data.xfrc_applied[TORSO_BODY_ID, :] = 0.0

        if jitter_std is not None:
            if current_action is None or steps_since_update >= next_hold_duration:
                current_action = policy(obs)
                steps_since_update = 0
                next_hold_duration = max(1, int(round(1 + np.random.normal(0, jitter_std))))
            action = current_action
            steps_since_update += 1
        else:
            action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            terminated_step = step
            break
        if truncated:
            break

    env.close()

    if terminated_step is None:
        return {"status": "recovered", "fell_at_step": None, "steps_after_push": None}

    if terminated_step < push_end_step:
        # Fell before the push even finished -- unrelated to perturbation,
        # discard rather than misreport as a perturbation failure.
        return {"status": "invalid", "fell_at_step": terminated_step, "steps_after_push": None}

    steps_after_push = terminated_step - push_end_step
    if steps_after_push > RECOVERY_WINDOW:
        # Fell, but long after the push -- likely an unrelated spontaneous
        # fall (this policy is known to occasionally fail late for
        # reasons unconnected to any perturbation), so treat as recovered.
        status = "recovered"
    else:
        status = "fell"

    return {"status": status, "fell_at_step": terminated_step, "steps_after_push": steps_after_push}


def run_combo_sweep(policy, label, setup_fn, push_magnitudes, push_axis, axis_name,
                     jitter_std=None, n_trials=1, body_weight_n=None):
    print(f"\n=== {label} -- {axis_name} push ===")
    if n_trials > 1:
        print("push_N, push_%BW, n_valid, n_recovered, n_fell, recovery_rate, mean_steps_after_fall")
    else:
        print("push_magnitude, status, fell_at_step, steps_after_push")

    for push in push_magnitudes:
        results = []
        for trial in range(n_trials):
            seed = trial if jitter_std is not None else None
            r = run_perturbation_trial(policy, setup_fn, push, push_axis,
                                         jitter_std=jitter_std, seed=seed)
            results.append(r)

        valid = [r for r in results if r["status"] != "invalid"]
        n_invalid = len(results) - len(valid)
        n_recovered = sum(1 for r in valid if r["status"] == "recovered")
        n_fell = sum(1 for r in valid if r["status"] == "fell")
        fell_steps = [r["steps_after_push"] for r in valid if r["status"] == "fell"]
        mean_fell_steps = np.mean(fell_steps) if fell_steps else float('nan')

        if n_trials == 1:
            r = results[0]
            print(f"{push}, {r['status']}, {r['fell_at_step']}, {r['steps_after_push']}")
        else:
            recovery_rate = n_recovered / len(valid) if len(valid) > 0 else float('nan')
            pct_bw = (push / body_weight_n * 100) if body_weight_n else float('nan')
            print(f"{push}, {pct_bw:.1f}%, {len(valid)}/{n_trials} valid "
                  f"({n_invalid} discarded), {n_recovered}, {n_fell}, "
                  f"{recovery_rate:.2f}, {mean_fell_steps:.1f}")


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    body_weight_n = get_body_weight_newtons(env.unwrapped.sim)
    env.close()

    print(f"Estimated body weight: {body_weight_n:.1f} N")

    push_magnitudes = [400, 500, 550, 600, 650, 700, 800]

    test_configs = [
        ("Healthy (0.85 strength)", lambda sim: apply_healthy_or_als(sim, 0.85), None, 1),
        ("ALS-inspired (0.6 strength)", lambda sim: apply_healthy_or_als(sim, 0.6), None, 1),
        ("Stroke-inspired (0.6 paretic, left)", lambda sim: apply_stroke(sim, 0.6, "left"), None, 1),
        ("Feedback-delay-inspired (jitter=1.0)", None, 1.0, 45),
    ]

    for axis, axis_name in [(SAGITTAL_AXIS, "SAGITTAL (forward/backward, along travel)"),
                              (LATERAL_AXIS, "LATERAL (sideways, perpendicular to travel)")]:
        for label, setup_fn, jitter, n_trials in test_configs:
            run_combo_sweep(policy, label, setup_fn, push_magnitudes, axis, axis_name,
                             jitter_std=jitter, n_trials=n_trials, body_weight_n=body_weight_n)
            
    print("\n\n=== FOLLOW-UP: locating the real stroke-combo threshold (lower force range) ===")
    low_push_magnitudes = [50, 100, 150, 200, 250, 300, 350]
    for axis, axis_name in [(SAGITTAL_AXIS, "SAGITTAL (forward/backward, along travel)"),
                              (LATERAL_AXIS, "LATERAL (sideways, perpendicular to travel)")]:
        run_combo_sweep(policy, "Stroke-inspired (0.6 paretic, left)",
                         lambda sim: apply_stroke(sim, 0.6, "left"),
                         low_push_magnitudes, axis, axis_name,
                         jitter_std=None, n_trials=1, body_weight_n=body_weight_n)