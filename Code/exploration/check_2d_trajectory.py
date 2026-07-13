from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
policy = deprl.load_baseline(env)
sim = env.unwrapped.sim

obs, _ = env.reset()
print("Tracking full position + orientation every 10 steps:")
print(f"step 0: x={sim.data.qpos[0]:.4f}, y={sim.data.qpos[1]:.4f}, z={sim.data.qpos[2]:.4f}")

for step in range(1, 117):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        x, y, z = sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2]
        total_dist = np.sqrt(x**2 + y**2)
        print(f"step {step}: x={x:.4f}, y={y:.4f}, z={z:.4f}, total_planar_dist={total_dist:.4f}")
    if terminated or truncated:
        x, y, z = sim.data.qpos[0], sim.data.qpos[1], sim.data.qpos[2]
        print(f"Terminated at step {step}: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        break

env.close()