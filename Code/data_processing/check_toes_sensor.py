from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()
l_foot_vals, l_toes_vals, r_foot_vals, r_toes_vals = [], [], [], []

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    l_foot_vals.append(sim.data.sensor('l_foot').data[0])
    l_toes_vals.append(sim.data.sensor('l_toes').data[0])
    r_foot_vals.append(sim.data.sensor('r_foot').data[0])
    r_toes_vals.append(sim.data.sensor('r_toes').data[0])
    if terminated or truncated:
        print(f"Terminated at step {step}")
        break

env.close()
for name, vals in [("l_foot", l_foot_vals), ("l_toes", l_toes_vals), ("r_foot", r_foot_vals), ("r_toes", r_toes_vals)]:
    vals = np.array(vals)
    print(f"{name}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}, nonzero_count={np.sum(vals != 0)}")