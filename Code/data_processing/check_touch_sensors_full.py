from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()
r_vals, l_vals = [], []

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r_vals.append(sim.data.sensor('r_foot').data[0])
    l_vals.append(sim.data.sensor('l_foot').data[0])
    if terminated or truncated:
        print(f"Terminated at step {step}")
        break

r_vals = np.array(r_vals)
l_vals = np.array(l_vals)

print(f"\nRight foot sensor: min={r_vals.min():.4f}, max={r_vals.max():.4f}, mean={r_vals.mean():.4f}, nonzero_count={np.sum(r_vals != 0)}")
print(f"Left foot sensor: min={l_vals.min():.4f}, max={l_vals.max():.4f}, mean={l_vals.mean():.4f}, nonzero_count={np.sum(l_vals != 0)}")

env.close()