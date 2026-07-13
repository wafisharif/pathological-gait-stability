from myosuite.utils import gym
import deprl
import numpy as np

STRENGTH_FACTOR = 0.6  # known to survive the full episode

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim

original_gainprm = sim.model.actuator_gainprm.copy()
scaled_gainprm = original_gainprm.copy()
scaled_gainprm[:, 2] = original_gainprm[:, 2] * STRENGTH_FACTOR
sim.model.actuator_gainprm[:] = scaled_gainprm

policy = deprl.load_baseline(env)
obs, _ = env.reset()

warmup_steps = 100
post_warmup_heights = []

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if step >= warmup_steps:
        obs_dict = env.unwrapped.get_obs_dict(sim)
        post_warmup_heights.append(np.array(obs_dict["feet_heights"]).copy())

    if terminated or truncated:
        print(f"Fell at step {step}")
        break

post_warmup_heights = np.array(post_warmup_heights)
print(f"Mean foot heights AFTER warmup, strength={STRENGTH_FACTOR} (R, L):", np.mean(post_warmup_heights, axis=0))
print(f"Number of post-warmup steps measured: {len(post_warmup_heights)}")

env.close()