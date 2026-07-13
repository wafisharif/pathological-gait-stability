from myosuite.utils import gym
import numpy as np

def run_episode(weakness_factor=1.0, max_steps=500, render=False):
    """
    weakness_factor: 1.0 = full strength (baseline), 
                      0.5 = 50% strength, etc.
    """
    env = gym.make('myoLegWalk-v0')
    obs = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        # Sample a baseline action (random for now -- later we'll swap this
        # for a real walking policy/controller, this is just to test the
        # weakness mechanism itself)
        action = env.action_space.sample()

        # Apply the weakness: scale every muscle activation down
        weakened_action = action * weakness_factor

        obs, reward, terminated, truncated, info = env.step(weakened_action)
        total_reward += reward
        steps_survived += 1

        if terminated or truncated:
            break

    env.close()
    return steps_survived, total_reward


if __name__ == "__main__":
    print("=== BASELINE (full strength) ===")
    steps, reward = run_episode(weakness_factor=1.0)
    print(f"Steps survived: {steps}, Total reward: {reward:.2f}")

    print("\n=== WEAKENED (50% strength) ===")
    steps, reward = run_episode(weakness_factor=0.5)
    print(f"Steps survived: {steps}, Total reward: {reward:.2f}")

    print("\n=== SEVERELY WEAKENED (20% strength) ===")
    steps, reward = run_episode(weakness_factor=0.2)
    print(f"Steps survived: {steps}, Total reward: {reward:.2f}")