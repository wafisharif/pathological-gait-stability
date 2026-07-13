from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()

print("Available sensor names:")
for i in range(sim.model.nsensor):
    print(i, sim.model.sensor(i).name)

print("\n=== Sample sensor readings over 20 steps ===")
for step in range(20):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    try:
        r_touch = sim.data.sensor('r_foot').data[0]
        l_touch = sim.data.sensor('l_foot').data[0]
        print(f"step {step}: r_foot={r_touch:.3f}, l_foot={l_touch:.3f}")
    except Exception as e:
        print(f"Error reading sensor: {e}")
        break
    if terminated or truncated:
        break

env.close()