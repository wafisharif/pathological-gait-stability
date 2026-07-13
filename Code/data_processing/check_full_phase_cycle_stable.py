"""
Rerun of the phase-cycle check, using strength=0.85 -- our confirmed,
validated, full-1000-step-survival configuration -- instead of the raw
baseline, which falls around step 116 due to the asymmetric starting
pose diagnosed early in this project. This avoids contaminating the
comparison with fall-related motion, and is a required check before
building any reward-function code for Phase 1 (physiologically-grounded
cost function).
"""
from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()

hip_angles_deg = []
phase_vars = []

for step in range(250):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    hip_raw = sim.data.joint('hip_flexion_r').qpos[0]
    obs_dict = env.unwrapped.get_obs_dict(sim)
    phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

    hip_angles_deg.append(np.degrees(hip_raw))
    phase_vars.append(phase)

    if terminated or truncated:
        print(f"Terminated at step {step} -- unexpected for strength=0.85")
        break

env.close()

print("=== Full trace, strength=0.85 (known stable), every 5 steps ===")
for i in range(0, len(phase_vars), 5):
    print(f"step {i}: phase_var={phase_vars[i]:.3f}, hip_angle_deg={hip_angles_deg[i]:.2f}")

print(f"\nTotal steps survived in this trace: {len(phase_vars)}")