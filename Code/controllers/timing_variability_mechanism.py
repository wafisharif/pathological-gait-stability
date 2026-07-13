"""
Timing variability mechanism -- NOT "the Parkinson's controller."
This mechanism introduces irregular timing in when muscle commands are
updated (jittery decision rate), representing the rhythm-irregularity
feature commonly observed in Parkinson's and Huntington's gait. This is
a control-timing / activation-level phenomenon, distinct from strength
(a mechanical muscle-capacity property).
"""
from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os


def run_episode_jittery_timing(policy, jitter_std, max_steps=1000, seed=None):
    """
    jitter_std: standard deviation (in simulation steps) of how long each
    action is held before requesting a new one from the policy. jitter_std=0
    means normal behavior (new action every step). Higher jitter_std means
    more irregular timing -- sometimes updating almost every step,
    sometimes lagging for several steps.
    """
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    feet_heights_trace = []
    steps_survived = 0
    current_action = None
    steps_since_update = 0
    next_hold_duration = 1

    for step in range(max_steps):
        if current_action is None or steps_since_update >= next_hold_duration:
            current_action = policy(obs)
            steps_since_update = 0
            # Sample a new hold duration -- always at least 1 step,
            # jittered around 1 by jitter_std, never negative.
            next_hold_duration = max(1, int(round(1 + np.random.normal(0, jitter_std))))

        obs, reward, terminated, truncated, info = env.step(current_action)
        steps_since_update += 1

        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        feet_heights_trace.append(np.array(obs_dict["feet_heights"]).copy())

        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    return np.array(feet_heights_trace), steps_survived


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

    jitter_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    n_episodes_per_level = 30  # pooling across episodes, same validated approach as motor noise

    results_path = "Results/timing_variability_pooled.csv"
    header = ["jitter_std", "n_episodes", "n_total_strikes", "mean_dur", "cv_dur"]

    print("Timing variability (jittery decision rate) -- pooled across episodes\n")
    print("jitter_std, n_episodes, n_strikes, mean_dur, cv_dur")
    for jitter in jitter_levels:
        pooled_durations = []
        for seed in range(n_episodes_per_level):
            feet_trace, steps = run_episode_jittery_timing(policy, jitter_std=jitter, seed=seed)
            pooled_durations.extend(get_strike_durations(feet_trace, foot_index=0))
            pooled_durations.extend(get_strike_durations(feet_trace, foot_index=1))

        if len(pooled_durations) >= 2:
            mean_d = np.mean(pooled_durations)
            std_d = np.std(pooled_durations)
            cv_d = std_d / mean_d if mean_d > 0 else float('nan')
        else:
            mean_d, cv_d = float('nan'), float('nan')

        print(f"{jitter}, {n_episodes_per_level}, {len(pooled_durations)}, {mean_d:.2f}, {cv_d:.3f}")
        log_result(results_path, [jitter, n_episodes_per_level, len(pooled_durations), mean_d, cv_d], header)