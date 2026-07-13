from myosuite.utils import gym
import deprl
from deprl import env_wrappers
import numpy as np

env = gym.make('myoLegWalk-v0')
env = env_wrappers.GymWrapper(env)
policy = deprl.load_baseline(env)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

sim = env.unwrapped.sim
print(f"step 0: x={sim.data.qpos[0]:.4f}")

for step in range(1, 117):
    action = policy(obs)
    result = env.step(action)
    obs = result[0]
    terminated = result[2] if len(result) > 2 else False
    if step % 10 == 0:
        print(f"step {step}: x={sim.data.qpos[0]:.4f}")
    if terminated:
        print(f"Terminated at step {step}, final x={sim.data.qpos[0]:.4f}")
        break

env.close()