from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

TRANSIENT_STEPS = 30  # skip these as "still recovering from start pose"

def run_and_measure(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    heights_post_transient = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_survived += 1

        if step >= TRANSIENT_STEPS:
            obs_dict = env.unwrapped.get_obs_dict(sim)
            heights_post_transient.append(np.array(obs_dict["feet_heights"]).copy())

        if terminated or truncated:
            break

    env.close()

    if len(heights_post_transient) < 5:
        return steps_survived, float('nan'), float('nan'), len(heights_post_transient)

    heights_post_transient = np.array(heights_post_transient)
    mean_r, mean_l = np.mean(heights_post_transient, axis=0)
    asymmetry = abs(mean_r - mean_l)
    return steps_survived, mean_r, mean_l, len(heights_post_transient), asymmetry


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

    strength_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2]
    results_path = "Results/strength_symmetry_sweep.csv"
    header = ["strength_factor", "steps_survived", "mean_foot_R", "mean_foot_L", "n_post_transient_steps", "asymmetry"]

    print("strength, steps_survived, mean_R, mean_L, n_usable_steps, asymmetry")
    for s in strength_levels:
        result = run_and_measure(policy, strength_factor=s)
        if len(result) == 5:
            steps, mean_r, mean_l, n_usable, asym = result
            print(f"{s}, {steps}, {mean_r:.3f}, {mean_l:.3f}, {n_usable}, {asym:.3f}")
            log_result(results_path, [s, steps, mean_r, mean_l, n_usable, asym], header)
        else:
            steps, mean_r, mean_l, n_usable = result
            print(f"{s}, {steps}, nan, nan, {n_usable}, nan (too few usable steps)")
            log_result(results_path, [s, steps, mean_r, mean_l, n_usable, float('nan')], header)