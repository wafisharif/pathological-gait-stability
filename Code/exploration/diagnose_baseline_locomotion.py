from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
policy = deprl.load_baseline(env)
sim = env.unwrapped.sim

obs, _ = env.reset()
print("Tracking x-position every 10 steps (unmodified, full-strength baseline):")
print(f"step 0: x={sim.data.qpos[0]:.4f}")

for step in range(1, 117):  # we know it falls around step 116
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        print(f"step {step}: x={sim.data.qpos[0]:.4f}")
    if terminated or truncated:
        print(f"Terminated at step {step}, final x={sim.data.qpos[0]:.4f}")
        break

env.close()