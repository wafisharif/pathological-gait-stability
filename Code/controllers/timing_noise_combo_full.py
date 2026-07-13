"""
Re-run of the timing-jitter + motor-noise combo (C/D), extended to also
measure double-support % and per-side stance %, so it has the same full
feature set as Combo A and B for the unified PCA dataset.
"""
from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

CONTACT_THRESHOLD = 10.0


def run_episode_combined(policy, jitter_std, noise_std, max_steps=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    r_trace, l_trace = [], []
    steps_survived = 0
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

        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return np.array(r_trace), np.array(l_trace), steps_survived, distance


def analyze_episode(r_trace, l_trace, steps):
    transient = min(30, steps // 4)
    r, l = r_trace[transient:], l_trace[transient:]
    n = len(r)
    if n < 10:
        return None

    r_on, l_on = r > CONTACT_THRESHOLD, l > CONTACT_THRESHOLD
    both = r_on & l_on

    r_strikes_idx = [i for i in range(1, len(r_on)) if r_on[i] and not r_on[i-1]]
    l_strikes_idx = [i for i in range(1, len(l_on)) if l_on[i] and not l_on[i-1]]

    durations = []
    if len(r_strikes_idx) >= 2:
        durations.extend(np.diff(r_strikes_idx))
    if len(l_strikes_idx) >= 2:
        durations.extend(np.diff(l_strikes_idx))

    return {
        "n_strikes": len(r_strikes_idx) + len(l_strikes_idx),
        "durations": durations,
        "double_support_pct": np.sum(both) / n * 100,
        "r_stance_pct": np.sum(r_on) / n * 100,
        "l_stance_pct": np.sum(l_on) / n * 100,
    }


def log_result(filepath, row, header):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    jitter_levels = [0.5, 1.0, 1.5]
    noise_levels  = [0.0, 0.05, 0.1, 0.2]
    n_episodes_per_combo = 50

    results_path = "Results/timing_noise_combo_full.csv"
    header = ["combo_name", "param1_name", "param1_val", "param2_name", "param2_val",
               "n_strikes", "mean_stride_dur", "cv_stride_dur",
               "double_support_pct", "r_stance_pct", "l_stance_pct",
               "steps_survived", "distance", "confidence"]

    print("Combo C/D (timing jitter + motor noise), extended with double-support/stance%\n")
    for jitter in jitter_levels:
        for noise in noise_levels:
            pooled_durations = []
            ds_list, r_stance_list, l_stance_list = [], [], []
            steps_list, dist_list = [], []
            total_strikes = 0

            for seed in range(n_episodes_per_combo):
                r_trace, l_trace, steps, dist = run_episode_combined(
                    policy, jitter_std=jitter, noise_std=noise, seed=seed
                )
                result = analyze_episode(r_trace, l_trace, steps)
                steps_list.append(steps)
                dist_list.append(dist)
                if result is not None:
                    pooled_durations.extend(result["durations"])
                    total_strikes += result["n_strikes"]
                    ds_list.append(result["double_support_pct"])
                    r_stance_list.append(result["r_stance_pct"])
                    l_stance_list.append(result["l_stance_pct"])

            if len(pooled_durations) >= 10:
                mean_d = np.mean(pooled_durations)
                cv_d = np.std(pooled_durations) / mean_d if mean_d > 0 else float('nan')
                confidence = "ok"
            elif len(pooled_durations) >= 2:
                mean_d = np.mean(pooled_durations)
                cv_d = np.std(pooled_durations) / mean_d if mean_d > 0 else float('nan')
                confidence = "low_confidence"
            else:
                mean_d, cv_d = float('nan'), float('nan')
                confidence = "too_short"

            mean_ds = np.mean(ds_list) if ds_list else float('nan')
            mean_r_stance = np.mean(r_stance_list) if r_stance_list else float('nan')
            mean_l_stance = np.mean(l_stance_list) if l_stance_list else float('nan')
            mean_steps = np.mean(steps_list)
            mean_dist = np.mean(dist_list)

            row = ["combo_cd_timing_noise", "jitter_std", jitter, "noise_std", noise,
                   total_strikes, mean_d, cv_d, mean_ds, mean_r_stance, mean_l_stance,
                   mean_steps, mean_dist, confidence]
            print(", ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in row))
            log_result(results_path, row, header)