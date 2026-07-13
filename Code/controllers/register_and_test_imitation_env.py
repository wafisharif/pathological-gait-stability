"""
Registers the new imitation-reward environment (WalkEnvV0Imitation) as
a usable gym environment ID, using the exact same model path and
settings as the real myoLegWalk-v0, then runs a short local sanity
test -- just confirming it loads and steps without crashing.
"""
import sys
import os
from myosuite.utils import gym
import numpy as np

# Import the custom environment class directly by file path, since our
# project folder isn't set up as a proper importable Python package
# (and has spaces in its path, which breaks standard module imports)
sys.path.insert(0, os.path.dirname(__file__))
from myoleg_walk_imitation_env import WalkEnvV0Imitation

MODEL_PATH = '/opt/anaconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/envs/myo/myobase/../../../simhive/myo_sim/leg/myolegs.xml'

# Register using a direct class reference instead of a string import path
gym.register(
    id='myoLegWalkImitation-v0',
    entry_point=WalkEnvV0Imitation,
    max_episode_steps=1000,
    kwargs={
        'model_path': MODEL_PATH,
        'normalize_act': True,
        'min_height': 0.8,
        'max_rot': 0.8,
        'hip_period': 100,
        'reset_type': 'init',
        'target_x_vel': 0.0,
        'target_y_vel': 1.2,
        'target_rot': None,
    }
)

print("Registered myoLegWalkImitation-v0 successfully.")

print("\nRunning local sanity test (20 random-action steps, no training)...")
env = gym.make('myoLegWalkImitation-v0')
obs, _ = env.reset()

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {step}: reward={reward:.4f}, terminated={terminated}")
    if terminated or truncated:
        print("Episode ended early (expected with random actions)")
        break

env.close()
print("\nSanity test complete -- no crashes means the environment and reward function are wired correctly.")