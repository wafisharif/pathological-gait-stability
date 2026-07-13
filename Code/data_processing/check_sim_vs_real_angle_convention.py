"""
Critical check before writing any reward-function code: does our
simulation's hip_flexion joint report values in the same units and
sign convention as the real motion-capture reference data? If not, we
need a conversion before comparing them in a reward term.
"""
from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()

hip_angles_raw = []
phase_vars = []

for step in range(300):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    hip_raw = sim.data.joint('hip_flexion_r').qpos[0]  # raw MuJoCo units
    obs_dict = env.unwrapped.get_obs_dict(sim)
    phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

    hip_angles_raw.append(hip_raw)
    phase_vars.append(phase)

    if terminated or truncated:
        break

env.close()

hip_angles_raw = np.array(hip_angles_raw)
hip_angles_deg = np.degrees(hip_angles_raw)

print("=== Raw sim hip_flexion_r values (radians) ===")
print(f"Range: {hip_angles_raw.min():.3f} to {hip_angles_raw.max():.3f}")

print("\n=== Converted to degrees ===")
print(f"Range: {hip_angles_deg.min():.2f} to {hip_angles_deg.max():.2f}")

print("\n=== Real reference data range (for comparison) ===")
print("HipAngles (real, degrees): -8.76 to 33.37")

print("\n=== First 20 raw values, converted to degrees, with phase_var ===")
for i in range(20):
    print(f"step {i}: phase_var={phase_vars[i]:.3f}, hip_angle_deg={hip_angles_deg[i]:.2f}")