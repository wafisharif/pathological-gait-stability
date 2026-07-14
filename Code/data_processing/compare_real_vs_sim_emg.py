"""
Checks whether real EMG data (from able-bodied motion capture) and our
simulation's own muscle activation values share a comparable scale and
shape, before building the muscle-timing reward term (part of the
originally-approved Phase 1 reward architecture, not yet built).
"""
from myosuite.utils import gym
import deprl
import numpy as np
import pandas as pd

# Real EMG muscle -> matching simulated muscle name
EMG_TO_SIM_MUSCLE = {
    "GASnorm": "gaslat_r",
    "RFnorm": "recfem_r",
    "VLnorm": "vaslat_r",
    "BFnorm": "bflh_r",
    "STnorm": "semiten_r",
    "TAnorm": "tibant_r",
}

# Load one real able-bodied subject's EMG curve as a first check
df = pd.read_excel(
    "Datasets/stroke_mocap/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx",
    sheet_name="Sub01"
)
print("Real EMG ranges (Sub01):")
for real_name in EMG_TO_SIM_MUSCLE:
    if real_name in df.columns:
        vals = df[real_name]
        print(f"  {real_name}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

# Now check our simulation's actual muscle activation values
env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()
muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]

activation_traces = {name: [] for name in EMG_TO_SIM_MUSCLE.values()}

for step in range(200):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    act = sim.data.act.copy()
    for sim_name in activation_traces:
        idx = muscle_names.index(sim_name)
        activation_traces[sim_name].append(act[idx])
    if terminated or truncated:
        break

env.close()

print("\nSimulated activation ranges (strength=0.85, 200 steps):")
for sim_name, trace in activation_traces.items():
    trace = np.array(trace)
    print(f"  {sim_name}: min={trace.min():.3f}, max={trace.max():.3f}, mean={trace.mean():.3f}")