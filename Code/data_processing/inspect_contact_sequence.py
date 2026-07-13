from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original_gainprm = sim.model.actuator_gainprm.copy()
scaled = original_gainprm.copy()
scaled[:, 2] = original_gainprm[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()
THRESHOLD = 10

print("step | R_force | L_force | R_on | L_on | pattern")
for step in range(150):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
    l = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]
    r_on = r > THRESHOLD
    l_on = l > THRESHOLD

    if step >= 30 and step < 100:  # skip transient, show a real walking chunk
        pattern = ("R" if r_on else "_") + ("L" if l_on else "_")
        print(f"{step:4d} | {r:7.1f} | {l:7.1f} | {str(r_on):5s} | {str(l_on):5s} | {pattern}")

    if terminated or truncated:
        break

env.close()