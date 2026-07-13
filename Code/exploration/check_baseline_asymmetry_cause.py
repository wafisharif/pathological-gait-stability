from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim

print("=== Checking initial pose symmetry at reset (before any policy action) ===")
obs, _ = env.reset()
obs_dict = env.unwrapped.get_obs_dict(sim)
print("Initial feet_heights (R, L):", obs_dict["feet_heights"])

print("\n=== Checking across multiple resets (is initial state always the same?) ===")
for trial in range(5):
    obs, _ = env.reset()
    obs_dict = env.unwrapped.get_obs_dict(sim)
    print(f"Reset {trial}: feet_heights (R, L) = {obs_dict['feet_heights']}")

env.close()