from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()
feet_heights = []

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    obs_dict = env.unwrapped.get_obs_dict(sim)
    feet_heights.append(np.array(obs_dict["feet_heights"]).copy())
    if terminated or truncated:
        break

env.close()
feet_heights = np.array(feet_heights)[30:]  # skip transient

for foot_idx, name in [(0, "Right"), (1, "Left")]:
    h = feet_heights[:, foot_idx]
    print(f"\n=== {name} foot height stats ===")
    print(f"min={np.min(h):.4f}, max={np.max(h):.4f}, mean={np.mean(h):.4f}")
    print("Percentage of time below various thresholds (as fraction of range):")
    for frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        threshold = np.min(h) + frac * (np.max(h) - np.min(h))
        pct_below = np.mean(h < threshold) * 100
        print(f"  threshold={frac} -> {pct_below:.1f}% of time below this")