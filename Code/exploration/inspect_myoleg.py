from myosuite.utils import gym
import numpy as np

env = gym.make('myoLegWalk-v0')
env.reset()

print("=== ACTION SPACE ===")
print(env.action_space)
print("Number of muscles (action dims):", env.action_space.shape)

print("\n=== OBSERVATION SPACE ===")
print(env.observation_space)
print("Obs vector size:", env.observation_space.shape)

print("\n=== MUSCLE / ACTUATOR NAMES ===")
sim = env.unwrapped.sim
muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
for i, name in enumerate(muscle_names):
    print(i, name)

print("\n=== OBSERVATION DICTIONARY KEYS ===")
obs_dict = env.unwrapped.get_obs_dict(sim)
for key, val in obs_dict.items():
    print(key, "-> shape:", np.shape(val))

print("\n=== REWARD DICTIONARY KEYS (quick check) ===")
try:
    reward_dict = env.unwrapped.get_reward_dict(obs_dict)
    for key, val in reward_dict.items():
        print(key, "->", val)
except Exception as e:
    print("Could not pull reward dict directly:", e)

print("\n=== DONE/TERMINATION CHECK ===")
print("Has 'done' logic accessible via env.unwrapped:", hasattr(env.unwrapped, "done"))

env.close()

print("\n=== TERMINATION CHECK (via step) ===")
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print("terminated:", terminated)
print("truncated:", truncated)