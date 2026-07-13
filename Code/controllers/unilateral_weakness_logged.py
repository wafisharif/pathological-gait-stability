from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

RIGHT_LEG_IDX = slice(0, 40)
LEFT_LEG_IDX = slice(40, 80)

def run_episode(policy, weak_side="left", weakness_factor=1.0, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs).copy()
        if weak_side == "left":
            action[LEFT_LEG_IDX] = action[LEFT_LEG_IDX] * weakness_factor
        elif weak_side == "right":
            action[RIGHT_LEG_IDX] = action[RIGHT_LEG_IDX] * weakness_factor

        obs, reward, terminated, truncated, info = env.step(action)
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

    weakness_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    results_path = "Results/controller_results.csv"
    header = ["controller_type", "param_name", "param_value", "side", "steps_survived", "total_reward"]

    for side in ["left", "right"]:
        print(f"=== WEAK SIDE: {side.upper()} ===")
        for w in weakness_levels:
            steps, reward = run_episode(policy, weak_side=side, weakness_factor=w)
            print(f"{w}, {steps}, {reward:.2f}")
            log_result(results_path, ["unilateral_weakness", "weakness_factor", w, side, steps, reward], header)