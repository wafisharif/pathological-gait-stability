"""
Extends the real-vs-simulated comparison to knee and ankle (pelvis
excluded here -- it's part of the free root body, not a simple hinge
joint, and needs a different extraction method to be checked correctly
rather than guessed).
"""
from myosuite.utils import gym
import deprl
import numpy as np

ref_data = np.load("Results/reference_trajectories.npz")

JOINTS_TO_CHECK = {
    "HipAngles": "hip_flexion_r",
    "KneeAngles": "knee_angle_r",
    "AnkleAngles": "ankle_angle_r",
}

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()

phase_trace = []
joint_traces = {name: [] for name in JOINTS_TO_CHECK}
cycle_started = False
cycle_count = 0

for step in range(600):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    obs_dict = env.unwrapped.get_obs_dict(sim)
    phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

    if not cycle_started and phase < 0.02:
        cycle_started = True

    if cycle_started:
        if len(phase_trace) > 0 and phase < phase_trace[-1] - 0.5:
            cycle_count += 1
            if cycle_count >= 2:
                break
        phase_trace.append(phase)
        for ref_name, mj_name in JOINTS_TO_CHECK.items():
            raw = sim.data.joint(mj_name).qpos[0]
            joint_traces[ref_name].append(np.degrees(raw))

    if terminated or truncated:
        break

env.close()

phase_trace = np.array(phase_trace)
print(f"Collected {len(phase_trace)} points across one full simulated gait cycle\n")

for ref_name in JOINTS_TO_CHECK:
    sim_trace = np.array(joint_traces[ref_name])
    real_curve = ref_data[f"{ref_name}_mean"]
    real_at_sim_phases = np.array([real_curve[int(p * 1000)] for p in phase_trace])

    sim_span = sim_trace.max() - sim_trace.min()
    real_span = real_at_sim_phases.max() - real_at_sim_phases.min()
    scale_ratio = sim_span / real_span
    mean_offset = np.mean(sim_trace - real_at_sim_phases)
    correlation = np.corrcoef(sim_trace, real_at_sim_phases)[0, 1]

    print(f"=== {ref_name} ===")
    print(f"  Sim range: {sim_trace.min():.2f} to {sim_trace.max():.2f} (span {sim_span:.2f})")
    print(f"  Real range: {real_at_sim_phases.min():.2f} to {real_at_sim_phases.max():.2f} (span {real_span:.2f})")
    print(f"  Scale ratio (sim/real): {scale_ratio:.2f}")
    print(f"  Mean offset (sim - real): {mean_offset:.2f}")
    print(f"  Correlation: {correlation:.3f}")
    print()