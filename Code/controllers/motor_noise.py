from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

def run_episode(policy, noise_std=0.0, max_steps=1000, seed=None):
    """
    noise_std: standard deviation of Gaussian noise added to each muscle
    command. 0.0 = no noise (clean baseline). Larger = more timing/motor
    variability, mapping to Parkinson's (lower noise) / Huntington's
    (higher noise) per our locked impairment mapping.
    """
    if seed is not None:
        np.random.seed(seed)

    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs).copy()
        noise = np.random.normal(loc=0.0, scale=noise_std, size=action.shape)
        noisy_action = action + noise
        # Clip back into valid range, since muscles only accept -1 to 1
        noisy_action = np.clip(noisy_action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(noisy_action)
        total_reward += reward
        steps_survived += 1

        if terminated or truncated:
            break

    env.close()
    return steps_survived, total_reward


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
    n_repeats = 3  # run multiple seeds since noise is now genuinely random

    results_path = "Results/controller_results.csv"
    header = ["controller_type", "param_name", "param_value", "seed", "steps_survived", "total_reward"]

    print("controller_type, param_value, seed, steps_survived, total_reward")
    for noise_std in noise_levels:
        for seed in range(n_repeats):
            steps, reward = run_episode(policy, noise_std=noise_std, seed=seed)
            print(f"motor_noise, {noise_std}, {seed}, {steps}, {reward:.2f}")
            log_result(results_path, ["motor_noise", "noise_std", noise_std, seed, steps, reward], header)