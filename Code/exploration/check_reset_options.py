from myosuite.utils import gym
import numpy as np

env = gym.make('myoLegWalk-v0')
unwrapped = env.unwrapped

print("=== Checking reset-related attributes ===")
for attr in ["init_qpos", "init_qvel", "reset_type", "target_x_vel"]:
    print(attr, ":", hasattr(unwrapped, attr))

print("\n=== sim.model qpos0 (default pose MuJoCo resets to) ===")
print(unwrapped.sim.model.qpos0)

print("\n=== Current qpos right after reset ===")
obs, _ = env.reset()
print(unwrapped.sim.data.qpos)

env.close()