from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

def run_episode_strength_scaled(policy, strength_factor=1.0, max_steps=1000):
    """
    strength_factor: 1.0 = full strength (baseline), 0.5 = 50% peak force, etc.
    This scales the actual muscle-tendon peak force parameter (gainprm index 2),
    NOT the activation/action signal -- this is the mechanically correct way
    to represent true weakness (ALS mapping), per Miles's correction.
    """
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    # Scale peak force for ALL muscles before the episode starts
    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled_gainprm = original_gainprm.copy()
    scaled_gainprm[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled_gainprm

    obs, _ = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)  # activation/action signal is UNCHANGED -- only strength is reduced
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

    strength_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    results_path = "Results/controller_strength_results.csv"
    header = ["controller_type", "param_name", "param_value", "side", "steps_survived", "total_reward"]

    print("strength_factor, steps_survived, total_reward")
    for s in strength_levels:
        steps, reward = run_episode_strength_scaled(policy, strength_factor=s)
        print(f"{s}, {steps}, {reward:.2f}")
        log_result(results_path, ["global_strength_weakness", "strength_factor", s, "both", steps, reward], header)