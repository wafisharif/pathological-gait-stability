from myosuite.utils import gym
import deprl

env = gym.make('myoLegWalk-v0')
policy = deprl.load_baseline(env)

obs, _ = env.reset()
total_reward = 0
steps_survived = 0

for i in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps_survived += 1
    if terminated or truncated:
        break

print(f"Steps survived: {steps_survived}")
print(f"Total reward: {total_reward:.2f}")

env.close()