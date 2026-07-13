"""
Real pelvis tilt barely varies (under 1 degree) across a real gait
cycle. If we extracted the correct axis/convention, our simulated
pelvis reading should ALSO be relatively stable (even if offset), not
swinging by itself. This checks all three rotation components to find
which one actually behaves like a stable "pelvis tilt" signal, rather
than assuming the second component (index 1) was correct.
"""
from myosuite.utils import gym
import deprl
import numpy as np
from myosuite.utils.quat_math import quat2euler

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()

all_axes_trace = {0: [], 1: [], 2: []}

for step in range(200):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    pelvis_quat = sim.data.body('pelvis').xquat.copy()
    pelvis_euler = quat2euler(pelvis_quat)

    for axis in range(3):
        all_axes_trace[axis].append(np.degrees(pelvis_euler[axis]))

    if terminated or truncated:
        break

env.close()

for axis in range(3):
    trace = np.array(all_axes_trace[axis])
    print(f"Axis {axis}: range {trace.min():.2f} to {trace.max():.2f}, "
          f"span {trace.max()-trace.min():.2f}, std {np.std(trace):.2f}")

print("\nReal pelvis tilt span was under 1 degree -- looking for whichever")
print("axis above has the SMALLEST span/std, since that's most likely the")
print("correct match to the real 'pelvis tilt' signal.")