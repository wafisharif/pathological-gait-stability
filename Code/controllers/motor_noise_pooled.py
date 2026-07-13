from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os


def run_episode_collect_strikes(policy, noise_std=0.0, max_steps=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    feet_heights_trace = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs).copy()
        noise = np.random.normal(loc=0.0, scale=noise_std, size=action.shape)
        noisy_action = np.clip(action + noise, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(noisy_action)
        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        feet_heights_trace.append(np.array(obs_dict["feet_heights"]).copy())

        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    return np.array(feet_heights_trace), steps_survived


def get_strike_durations(feet_heights_trace, foot_index=0):
    """Returns the list of step-to-step-strike durations for one episode
    (may be empty if too few strikes occurred)."""
    if feet_heights_trace.ndim != 2 or feet_heights_trace.shape[0] < 3:
        return []

    heights = feet_heights_trace[:, foot_index]
    height_range = np.max(heights) - np.min(heights)
    if height_range <= 1e-8:
        return []

    threshold = np.min(heights) + 0.25 * height_range
    on_ground = heights < threshold

    strike_indices = []
    for i in range(1, len(on_ground)):
        if on_ground[i] and not on_ground[i - 1]:
            strike_indices.append(i)

    if len(strike_indices) < 2:
        return []

    return list(np.diff(strike_indices))


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

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    n_episodes_per_level = 50  # pool across many short episodes instead of relying on 1 long one

    results_path = "Results/controller_motor_noise_pooled.csv"
    header = ["controller_type", "param_value", "n_episodes", "n_total_strikes",
               "mean_dur_right", "cv_dur_right", "mean_dur_left", "cv_dur_left"]

    print("noise_std, n_episodes, n_strikes_R, cv_dur_R, n_strikes_L, cv_dur_L")
    for noise_std in noise_levels:
        pooled_right = []
        pooled_left = []

        for seed in range(n_episodes_per_level):
            feet_trace, steps = run_episode_collect_strikes(policy, noise_std=noise_std, seed=seed)
            pooled_right.extend(get_strike_durations(feet_trace, foot_index=0))
            pooled_left.extend(get_strike_durations(feet_trace, foot_index=1))

        def summarize(durations):
            if len(durations) < 2:
                return float('nan'), float('nan'), len(durations)
            mean_d = np.mean(durations)
            std_d = np.std(durations)
            cv_d = std_d / mean_d if mean_d > 0 else float('nan')
            return mean_d, cv_d, len(durations)

        mean_r, cv_r, n_r = summarize(pooled_right)
        mean_l, cv_l, n_l = summarize(pooled_left)

        print(f"{noise_std}, {n_episodes_per_level}, {n_r}, {cv_r:.3f}, {n_l}, {cv_l:.3f}")
        log_result(results_path,
                    ["motor_noise_pooled", noise_std, n_episodes_per_level, n_r + n_l, mean_r, cv_r, mean_l, cv_l],
                    header)