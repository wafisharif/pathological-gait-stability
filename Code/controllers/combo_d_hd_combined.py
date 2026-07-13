"""
Combo D: Huntington's-inspired combined mechanism -- global strength/vmax
reduction (general motor slowing) PLUS feedback-delay/motor-noise
(chorea-like timing irregularity), tested together for the first time.
Real Huntington's patients show both dramatic overall slowing (stride
time ~2x healthy) AND high timing variability -- a single mechanism
category (tested in isolation, see find_real_hd_config.py) could not
reach this combined signature. This tests whether combining mechanisms,
as Miles suggested, closes that gap.
"""
from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

CONTACT_THRESHOLD = 10.0


def run_episode(policy, strength, vmax, jitter_std, noise_std, max_steps=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    scaled[:, 2] = original[:, 2] * strength
    scaled[:, 6] = original[:, 6] * vmax
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    r_trace, l_trace = [], []
    steps = 0
    current_action = None
    steps_since_update = 0
    next_hold_duration = 1

    for step in range(max_steps):
        if current_action is None or steps_since_update >= next_hold_duration:
            base_action = policy(obs)
            noise = np.random.normal(loc=0.0, scale=noise_std, size=base_action.shape)
            current_action = np.clip(base_action + noise, -1.0, 1.0)
            steps_since_update = 0
            next_hold_duration = max(1, int(round(1 + np.random.normal(0, jitter_std))))

        obs, reward, terminated, truncated, info = env.step(current_action)
        steps_since_update += 1

        r_trace.append(sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0])
        l_trace.append(sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0])
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return np.array(r_trace), np.array(l_trace), steps


def analyze_episode(r_trace, l_trace, steps):
    transient = min(30, steps // 4)
    r, l = r_trace[transient:], l_trace[transient:]
    n = len(r)
    if n < 10:
        return None

    r_on, l_on = r > CONTACT_THRESHOLD, l > CONTACT_THRESHOLD
    both = r_on & l_on

    r_strikes = [i for i in range(1, len(r_on)) if r_on[i] and not r_on[i-1]]
    l_strikes = [i for i in range(1, len(l_on)) if l_on[i] and not l_on[i-1]]

    durations = []
    if len(r_strikes) >= 2:
        durations.extend(np.diff(r_strikes))
    if len(l_strikes) >= 2:
        durations.extend(np.diff(l_strikes))

    return {
        "n_strikes": len(r_strikes) + len(l_strikes),
        "durations": durations,
        "double_support_pct": np.sum(both) / n * 100,
        "r_stance_pct": np.sum(r_on) / n * 100,
        "l_stance_pct": np.sum(l_on) / n * 100,
    }


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    # Combine our known-working strength/vmax severity (from Combo A) with
    # our known-working jitter/noise severity (from Combo C)
    configs = [
        (0.85, 1.0, 0.5, 0.0),   # mild everything
        (0.75, 0.9, 1.0, 0.05),  # moderate
        (0.6, 0.8, 1.5, 0.1),    # more severe
        (0.6, 0.7, 1.5, 0.2),    # most severe combined
    ]

    n_episodes = 30
    results_path = "Results/combo_d_hd_combined.csv"
    header = ["strength", "vmax", "jitter", "noise", "n_strikes", "mean_stride_dur_s",
               "cv_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct", "confidence"]

    print("strength, vmax, jitter, noise, n_strikes, mean_dur_s, cv_dur, ds%, r_stance%, l_stance%, conf")
    for strength, vmax, jitter, noise in configs:
        pooled_durations = []
        ds_list, r_stance_list, l_stance_list = [], [], []

        for seed in range(n_episodes):
            r_trace, l_trace, steps = run_episode(policy, strength, vmax, jitter, noise, seed=seed)
            result = analyze_episode(r_trace, l_trace, steps)
            if result is not None:
                pooled_durations.extend(result["durations"])
                ds_list.append(result["double_support_pct"])
                r_stance_list.append(result["r_stance_pct"])
                l_stance_list.append(result["l_stance_pct"])

        if len(pooled_durations) >= 10:
            mean_dur_steps = np.mean(pooled_durations)
            mean_dur_s = mean_dur_steps * 0.01
            cv_dur = np.std(pooled_durations) / mean_dur_steps
            confidence = "ok"
        else:
            mean_dur_s, cv_dur = float('nan'), float('nan')
            confidence = "low_confidence"

        mean_ds = np.mean(ds_list) if ds_list else float('nan')
        mean_r = np.mean(r_stance_list) if r_stance_list else float('nan')
        mean_l = np.mean(l_stance_list) if l_stance_list else float('nan')

        row = [strength, vmax, jitter, noise, len(pooled_durations), mean_dur_s, cv_dur,
               mean_ds, mean_r, mean_l, confidence]
        print(", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in row))

        file_exists = os.path.isfile(results_path)
        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)