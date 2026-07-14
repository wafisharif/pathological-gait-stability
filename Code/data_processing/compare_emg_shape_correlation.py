"""
Checks EMG timing/shape correlation (not just average level) between
real and simulated muscle activation, using the same phase-matching
method already validated for joint angles.
"""
from myosuite.utils import gym
import deprl
import numpy as np
import pandas as pd

EMG_TO_SIM_MUSCLE = {
    "GASnorm": "gaslat_r", "RFnorm": "recfem_r", "VLnorm": "vaslat_r",
    "BFnorm": "bflh_r", "STnorm": "semiten_r", "TAnorm": "tibant_r",
}

# Build averaged real EMG reference curves across 30 subjects (same
# approach as build_reference_trajectories.py, applied to EMG this time)
xls = pd.ExcelFile("Datasets/stroke_mocap/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx")
subject_sheets = [s for s in xls.sheet_names if s.startswith("Sub")][:30]

real_curves = {name: [] for name in EMG_TO_SIM_MUSCLE}
for sheet in subject_sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    for real_name in EMG_TO_SIM_MUSCLE:
        if real_name in df.columns:
            real_curves[real_name].append(df[real_name].values)

for name in real_curves:
    stacked = np.array(real_curves[name])
    n_valid_subjects = np.sum(~np.isnan(stacked).all(axis=1))
    print(f"{name}: {n_valid_subjects} / {len(stacked)} subjects have valid data")
    real_curves[name] = np.nanmean(stacked, axis=0)

# Collect one full simulated gait cycle, same method as joint-angle checks
env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()
muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]

phase_trace = []
activation_traces = {name: [] for name in EMG_TO_SIM_MUSCLE.values()}
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
        act = sim.data.act.copy()
        for sim_name in activation_traces:
            idx = muscle_names.index(sim_name)
            activation_traces[sim_name].append(act[idx])

    if terminated or truncated:
        break

env.close()
phase_trace = np.array(phase_trace)

print(f"Collected {len(phase_trace)} points across one full simulated gait cycle\n")

for real_name, sim_name in EMG_TO_SIM_MUSCLE.items():
    sim_trace = np.array(activation_traces[sim_name])
    real_curve = real_curves[real_name]
    real_at_sim_phases = np.array([real_curve[int(p * 1000)] for p in phase_trace])

    print(f"DEBUG {real_name}: sim_trace shape={sim_trace.shape}, real_at_sim_phases shape={real_at_sim_phases.shape}")
    print(f"DEBUG {real_name}: sim_trace has NaN={np.isnan(sim_trace).any()}, real has NaN={np.isnan(real_at_sim_phases).any()}")
    print(f"DEBUG {real_name}: sim_trace std={np.std(sim_trace):.6f}, real std={np.std(real_at_sim_phases):.6f}")

    correlation = np.corrcoef(sim_trace, real_at_sim_phases)[0, 1]
    mean_ratio = sim_trace.mean() / real_at_sim_phases.mean() if real_at_sim_phases.mean() > 0 else float('nan')

    print(f"{real_name} / {sim_name}: correlation={correlation:.3f}, mean_ratio(sim/real)={mean_ratio:.2f}")