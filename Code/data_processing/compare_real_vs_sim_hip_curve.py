"""
Direct, phase-point-by-phase-point comparison of the real reference hip
curve (from able-bodied motion capture) against our simulated hip curve
(strength=0.85, known stable). Determines whether the mismatch is a
constant offset, a scale difference, or a shape difference, so we know
exactly what correction the Phase 1 reward function needs.
"""
from myosuite.utils import gym
import deprl
import numpy as np

# Load the real reference curve built earlier
ref_data = np.load("Results/reference_trajectories.npz")
real_hip_curve = ref_data["HipAngles_mean"]  # shape (1001,), indexed 0-1000 by phase

# Re-run our stable sim config and collect a full phase-indexed hip trace
env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

original = sim.model.actuator_gainprm.copy()
scaled = original.copy()
scaled[:, 2] = original[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()

# Collect one full, clean cycle: from the first time phase_var is near 0,
# until it wraps back to near 0 again
phase_trace = []
hip_trace = []
cycle_started = False
cycle_count = 0

for step in range(600):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    hip_raw = sim.data.joint('hip_flexion_r').qpos[0]
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
        hip_trace.append(np.degrees(hip_raw))

    if terminated or truncated:
        break

env.close()

phase_trace = np.array(phase_trace)
hip_trace = np.array(hip_trace)

# Map each simulated phase point to the matching index (0-1000) in the real curve
real_at_sim_phases = np.array([real_hip_curve[int(p * 1000)] for p in phase_trace])

print(f"Collected {len(phase_trace)} points across one full simulated gait cycle\n")

print("=== Side-by-side comparison at 10 evenly-spaced phase points ===")
print(f"{'phase':>7} {'sim_deg':>10} {'real_deg':>10} {'diff':>8}")
step_size = max(1, len(phase_trace) // 10)
for i in range(0, len(phase_trace), step_size):
    diff = hip_trace[i] - real_at_sim_phases[i]
    print(f"{phase_trace[i]:>7.3f} {hip_trace[i]:>10.2f} {real_at_sim_phases[i]:>10.2f} {diff:>8.2f}")

print(f"\n=== Summary statistics ===")
print(f"Sim hip range: {hip_trace.min():.2f} to {hip_trace.max():.2f} (span: {hip_trace.max()-hip_trace.min():.2f})")
print(f"Real hip range: {real_at_sim_phases.min():.2f} to {real_at_sim_phases.max():.2f} (span: {real_at_sim_phases.max()-real_at_sim_phases.min():.2f})")
print(f"Mean difference (sim - real): {np.mean(hip_trace - real_at_sim_phases):.2f}")
print(f"Scale ratio (sim span / real span): {(hip_trace.max()-hip_trace.min()) / (real_at_sim_phases.max()-real_at_sim_phases.min()):.2f}")

correlation = np.corrcoef(hip_trace, real_at_sim_phases)[0, 1]
print(f"Correlation between sim and real shape (same phase points): {correlation:.3f}")
print("(High positive correlation = same SHAPE, just offset/scaled. Low/negative = actually different pattern.)")