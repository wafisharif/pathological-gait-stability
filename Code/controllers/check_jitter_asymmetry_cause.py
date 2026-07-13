from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()
np.random.seed(0)

current_action = None
steps_since_update = 0
next_hold_duration = 1

print("step | r_force | l_force | hold_duration")
for step in range(100):
    if current_action is None or steps_since_update >= next_hold_duration:
        current_action = policy(obs)
        steps_since_update = 0
        next_hold_duration = max(1, int(round(1 + np.random.normal(0, 1.0))))

    obs, reward, terminated, truncated, info = env.step(current_action)
    steps_since_update += 1

    r = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
    l = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]

    if step >= 30 and step % 3 == 0:
        print(f"{step} | {r:.1f} | {l:.1f} | {next_hold_duration}")

    if terminated or truncated:
        print(f"Terminated at step {step}")
        break

env.close()