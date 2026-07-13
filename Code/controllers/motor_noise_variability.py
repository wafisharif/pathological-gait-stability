from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os


def run_episode_record_data(policy, noise_std=0.0, max_steps=1000, seed=None):
    """
    Runs one episode, applying Gaussian noise to the policy's muscle
    commands (our motor-noise / timing-variability impairment mechanism,
    mapped to Parkinson's/Huntington's). Records feet_heights at every
    step, since that's real physical sensor data we can use to detect
    actual foot-strike events later.
    """
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    feet_heights_trace = []
    steps_survived = 0
    total_reward = 0

    for step in range(max_steps):
        action = policy(obs).copy()
        noise = np.random.normal(loc=0.0, scale=noise_std, size=action.shape)
        noisy_action = np.clip(action + noise, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(noisy_action)

        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        feet_heights_trace.append(np.array(obs_dict["feet_heights"]).copy())

        total_reward += reward
        steps_survived += 1

        if terminated or truncated:
            break

    env.close()
    return np.array(feet_heights_trace), steps_survived, total_reward


def compute_foot_strike_variability(feet_heights_trace, foot_index=0, height_threshold=None):
    """
    Detects foot-strike events (moments a foot transitions from
    swing/raised to ground contact) using real foot-height sensor data,
    and measures the time (in simulation steps) between consecutive
    strikes of the same foot. Higher variability (cv_dur) in this
    timing is the literature-aligned signature for Parkinson's/
    Huntington's-like motor noise -- NOT survival count, which is the
    wrong metric for this impairment type.

    foot_index: 0 = right foot, 1 = left foot (confirmed from MuJoCo
    site-name ordering: right-side sites/muscles listed first, then
    left-side -- see check_foot_site_names.py output).
    """
    feet_heights_trace = np.array(feet_heights_trace)

    if feet_heights_trace.ndim != 2 or feet_heights_trace.shape[0] < 3:
        return float('nan'), float('nan'), float('nan'), 0

    heights = feet_heights_trace[:, foot_index]

    if height_threshold is None:
        # Threshold relative to this episode's own height range, since
        # absolute foot height can shift depending on gait pattern.
        height_range = np.max(heights) - np.min(heights)
        if height_range <= 1e-8:
            return float('nan'), float('nan'), float('nan'), 0
        height_threshold = np.min(heights) + 0.25 * height_range

    on_ground = heights < height_threshold

    # Find rising edges of "on_ground" -- i.e. moments a foot just
    # touched down after being raised.
    strike_indices = []
    for i in range(1, len(on_ground)):
        if on_ground[i] and not on_ground[i - 1]:
            strike_indices.append(i)

    if len(strike_indices) < 2:
        return float('nan'), float('nan'), float('nan'), len(strike_indices)

    durations = np.diff(strike_indices)
    mean_dur = np.mean(durations)
    std_dur = np.std(durations)
    cv_dur = std_dur / mean_dur if mean_dur > 0 else float('nan')
    return mean_dur, std_dur, cv_dur, len(strike_indices)


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
    n_repeats = 3

    results_path = "Results/controller_foot_strike_results.csv"
    header = ["controller_type", "param_value", "seed", "steps_survived",
              "n_strikes_right", "mean_dur_right", "cv_dur_right",
              "n_strikes_left", "mean_dur_left", "cv_dur_left"]

    print("param_value, seed, steps_survived, n_strikes_R, cv_dur_R, n_strikes_L, cv_dur_L")
    for noise_std in noise_levels:
        for seed in range(n_repeats):
            feet_trace, steps, reward = run_episode_record_data(policy, noise_std=noise_std, seed=seed)

            mean_r, std_r, cv_r, n_r = compute_foot_strike_variability(feet_trace, foot_index=0)
            mean_l, std_l, cv_l, n_l = compute_foot_strike_variability(feet_trace, foot_index=1)

            print(f"{noise_std}, {seed}, {steps}, {n_r}, {cv_r:.3f}, {n_l}, {cv_l:.3f}")
            log_result(
                results_path,
                ["motor_noise", noise_std, seed, steps, n_r, mean_r, cv_r, n_l, mean_l, cv_l],
                header,
            )