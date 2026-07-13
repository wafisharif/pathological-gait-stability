from myosuite.utils import gym
import deprl
import numpy as np

# Confirmed earlier: indices 0-39 = right leg muscles, 40-79 = left leg muscles
RIGHT_LEG_IDX = slice(0, 40)
LEFT_LEG_IDX = slice(40, 80)

def run_episode(policy, weak_side="left", weakness_factor=1.0, max_steps=1000):
    """
    weak_side: "left" or "right" -- which leg gets weakened
    weakness_factor: 1.0 = no weakness, 0.5 = 50% strength on the weak side
    The other leg always stays at full strength (1.0).
    """
    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs).copy()  # copy so we don't modify the original array in place

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


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    weakness_levels = [1.0, 0.8, 0.6, 0.4, 0.2]

    print("=== WEAK SIDE: LEFT ===")
    print("weakness_factor, steps_survived, total_reward")
    for w in weakness_levels:
        steps, reward = run_episode(policy, weak_side="left", weakness_factor=w)
        print(f"{w}, {steps}, {reward:.2f}")

    print("\n=== WEAK SIDE: RIGHT ===")
    print("weakness_factor, steps_survived, total_reward")
    for w in weakness_levels:
        steps, reward = run_episode(policy, weak_side="right", weakness_factor=w)
        print(f"{w}, {steps}, {reward:.2f}")