from myosuite.utils import gym
import deprl
import numpy as np

def run_episode(policy, weakness_factor=1.0, max_steps=1000):
    """
    weakness_factor: 1.0 = full strength (baseline),
                      0.5 = 50% strength, etc.
    Applies uniform weakness to ALL muscles -- this is our
    global weakness mechanism (ALS mapping).
    """
    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    total_reward = 0
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        weakened_action = action * weakness_factor

        obs, reward, terminated, truncated, info = env.step(weakened_action)
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
    n_episodes_per_level = 5  # run several episodes, then average

    print("weakness_factor, avg_steps_survived, avg_total_reward")
    for w in weakness_levels:
        all_steps = []
        all_rewards = []
        for ep in range(n_episodes_per_level):
            steps, reward = run_episode(policy, weakness_factor=w)
            all_steps.append(steps)
            all_rewards.append(reward)
        avg_steps = sum(all_steps) / len(all_steps)
        avg_reward = sum(all_rewards) / len(all_rewards)
        print(f"{w}, {avg_steps:.1f}, {avg_reward:.2f}  (raw steps: {all_steps})")