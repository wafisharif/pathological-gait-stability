"""
Pelvis angle extraction and comparison -- last remaining joint needed
before the Phase 1 imitation reward function can be finalized. Pelvis
isn't a hinge joint like hip/knee/ankle; it comes from the free root
body's orientation quaternion, same conversion method already validated
when building the reflex controller earlier in this project.
"""
from myosuite.utils import gym
import deprl
import numpy as np
from myosuite.utils.quat_math import quat2euler

ref_data = np.load("Results/reference_trajectories.npz")
real_curve = ref_data["PelvisAngles_mean"]

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()

phase_trace = []
pelvis_trace = []
cycle_started = False
cycle_count = 0

for step in range(600):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    obs_dict = env.unwrapped.get_obs_dict(sim)
    phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

    pelvis_quat = sim.data.body('pelvis').xquat.copy()
    pelvis_euler = quat2euler(pelvis_quat)
    pelvis_angle_deg = np.degrees(pelvis_euler[1])  # sagittal-plane component, matching convention used for torso in reflex controller work

    if not cycle_started and phase < 0.02:
        cycle_started = True

    if cycle_started:
        if len(phase_trace) > 0 and phase < phase_trace[-1] - 0.5:
            cycle_count += 1
            if cycle_count >= 2:
                break
        phase_trace.append(phase)
        pelvis_trace.append(pelvis_angle_deg)

    if terminated or truncated:
        break

env.close()

phase_trace = np.array(phase_trace)
pelvis_trace = np.array(pelvis_trace)
real_at_sim_phases = np.array([real_curve[int(p * 1000)] for p in phase_trace])

sim_span = pelvis_trace.max() - pelvis_trace.min()
real_span = real_at_sim_phases.max() - real_at_sim_phases.min()

print(f"Collected {len(phase_trace)} points across one full simulated gait cycle\n")
print(f"Sim pelvis range: {pelvis_trace.min():.2f} to {pelvis_trace.max():.2f} (span {sim_span:.2f})")
print(f"Real pelvis range: {real_at_sim_phases.min():.2f} to {real_at_sim_phases.max():.2f} (span {real_span:.2f})")

if real_span > 1e-6:
    print(f"Scale ratio (sim/real): {sim_span / real_span:.2f}")
print(f"Mean offset (sim - real): {np.mean(pelvis_trace - real_at_sim_phases):.2f}")

if np.std(pelvis_trace) > 1e-6 and np.std(real_at_sim_phases) > 1e-6:
    correlation = np.corrcoef(pelvis_trace, real_at_sim_phases)[0, 1]
    print(f"Correlation: {correlation:.3f}")
else:
    print("Correlation: undefined (one of the signals is essentially flat)")