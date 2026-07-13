"""
Watches hip angle across one FULL phase_var cycle (0 to 1), not just a
small slice, to check whether hip motion completes exactly one
rise-and-fall per phase_var cycle (matching real gait data) or multiple
oscillations (which would mean phase_var doesn't track actual gait
rhythm the way we assumed).
"""
from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()

hip_angles_deg = []
phase_vars = []

for step in range(250):  # long enough to cover at least one full 0->1 phase cycle
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    hip_raw = sim.data.joint('hip_flexion_r').qpos[0]
    obs_dict = env.unwrapped.get_obs_dict(sim)
    phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

    hip_angles_deg.append(np.degrees(hip_raw))
    phase_vars.append(phase)

    if terminated or truncated:
        print(f"Terminated at step {step}")
        break

env.close()

print("=== Full trace: phase_var and hip angle together, every 5 steps ===")
for i in range(0, len(phase_vars), 5):
    print(f"step {i}: phase_var={phase_vars[i]:.3f}, hip_angle_deg={hip_angles_deg[i]:.2f}")

print(f"\nphase_var range covered: {min(phase_vars):.3f} to {max(phase_vars):.3f}")
print(f"Number of times phase_var wrapped from ~1.0 back to ~0.0: ", end="")
wraps = sum(1 for i in range(1, len(phase_vars)) if phase_vars[i] < phase_vars[i-1] - 0.5)
print(wraps)