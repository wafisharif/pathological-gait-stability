from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os


def get_strike_durations(heights, foot_index):
    h = heights[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return []
    threshold = np.min(h) + 0.25 * height_range
    on_ground = h < threshold
    strikes = [i for i in range(1, len(on_ground)) if on_ground[i] and not on_ground[i - 1]]
    if len(strikes) < 2:
        return []
    return list(np.diff(strikes))


def run_episode(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    feet_heights = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(sim)
        feet_heights.append(np.array(obs_dict["feet_heights"]).copy())
        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    return np.array(feet_heights), steps_survived


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

    strength_levels = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    results_path = "Results/als_strength_variability.csv"
    header = ["strength_factor", "steps_survived", "n_strikes_R", "cv_dur_R", "n_strikes_L", "cv_dur_L"]

    print("strength, steps, n_strikes_R, cv_dur_R, n_strikes_L, cv_dur_L")
    for s in strength_levels:
        feet_heights, steps = run_episode(policy, strength_factor=s)
        transient = min(30, steps // 4)
        feet_post = feet_heights[transient:]

        durations_r = get_strike_durations(feet_post, foot_index=0)
        durations_l = get_strike_durations(feet_post, foot_index=1)

        def summarize(durations):
            if len(durations) < 2:
                return float('nan'), len(durations)
            cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else float('nan')
            return cv, len(durations)

        cv_r, n_r = summarize(durations_r)
        cv_l, n_l = summarize(durations_l)

        print(f"{s}, {steps}, {n_r}, {cv_r:.3f}, {n_l}, {cv_l:.3f}")
        log_result(results_path, [s, steps, n_r, cv_r, n_l, cv_l], header)