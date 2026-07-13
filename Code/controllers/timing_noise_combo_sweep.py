"""
Combined timing-variability + motor-noise mechanism sweep.
Per Miles's feedback: this deliberately combines two mechanisms rather
than testing one in isolation, to avoid the "single mechanism explains
the whole pathology" oversimplification. Results are logged as full
feature vectors (not labeled with a specific disease name) -- the
comparison to real pathology profiles happens later, during the PCA step.
"""
from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os


def run_episode_combined(policy, jitter_std, noise_std, max_steps=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    feet_heights_trace = []
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

        obs_dict = env.unwrapped.get_obs_dict(sim)
        feet_heights_trace.append(np.array(obs_dict["feet_heights"]).copy())

        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return np.array(feet_heights_trace), steps_survived, distance


def get_strike_durations(feet_heights_trace, foot_index):
    if feet_heights_trace.ndim != 2 or feet_heights_trace.shape[0] < 3:
        return []
    h = feet_heights_trace[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return []
    threshold = np.min(h) + 0.25 * height_range
    on_ground = h < threshold
    strikes = [i for i in range(1, len(on_ground)) if on_ground[i] and not on_ground[i-1]]
    if len(strikes) < 2:
        return []
    return list(np.diff(strikes))


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

    jitter_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    noise_levels  = [0.0, 0.05, 0.1, 0.2, 0.3]
    n_episodes_per_combo = 15  # pooling, same validated approach as before

    results_path = "Results/timing_noise_combo_grid.csv"
    header = ["jitter_std", "noise_std", "n_episodes", "n_total_strikes",
               "mean_stride_dur", "cv_stride_dur", "mean_steps_survived", "mean_distance"]

    print("Combined timing-jitter + motor-noise grid sweep\n")
    print("jitter, noise, n_strikes, mean_dur, cv_dur, mean_steps, mean_dist")
    for jitter in jitter_levels:
        for noise in noise_levels:
            pooled_durations = []
            steps_list, dist_list = [], []

            for seed in range(n_episodes_per_combo):
                feet_trace, steps, dist = run_episode_combined(
                    policy, jitter_std=jitter, noise_std=noise, seed=seed
                )
                pooled_durations.extend(get_strike_durations(feet_trace, foot_index=0))
                pooled_durations.extend(get_strike_durations(feet_trace, foot_index=1))
                steps_list.append(steps)
                dist_list.append(dist)

            if len(pooled_durations) >= 2:
                mean_d = np.mean(pooled_durations)
                cv_d = np.std(pooled_durations) / mean_d if mean_d > 0 else float('nan')
            else:
                mean_d, cv_d = float('nan'), float('nan')

            mean_steps = np.mean(steps_list)
            mean_dist = np.mean(dist_list)

            print(f"{jitter}, {noise}, {len(pooled_durations)}, {mean_d:.2f}, {cv_d:.3f}, "
                  f"{mean_steps:.1f}, {mean_dist:.2f}")
            log_result(results_path,
                        [jitter, noise, n_episodes_per_combo, len(pooled_durations),
                         mean_d, cv_d, mean_steps, mean_dist],
                        header)