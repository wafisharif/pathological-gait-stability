"""
Extracts joint-angle ROM asymmetry from our simulated stroke combo
(unilateral strength + vmax weakening), using the same asymmetry formula
as the real stroke motion-capture analysis, for direct comparison.

Asymmetry formula (matches process_stroke_mocap.py):
    (nonparetic_ROM - paretic_ROM) / (paretic_ROM + nonparetic_ROM)
Negative = paretic side shows MORE range of motion (compensatory pattern
seen in real data for knee/hip). Positive = paretic side shows LESS ROM.
"""
from myosuite.utils import gym
import deprl
import numpy as np

# Our best "ok" confidence, full-survival stroke config from Combo B
BASE_STRENGTH = 0.6
PARETIC_VMAX = 1.0
PARETIC_SIDE = "left"  # matches our earlier stroke combo convention

JOINTS = {
    "AnkleAngles": "ankle_angle",
    "KneeAngles": "knee_angle",
    "HipAngles": "hip_flexion",
}


def run_stroke_combo(policy, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    idx = slice(40, 80) if PARETIC_SIDE == "left" else slice(0, 40)
    scaled[idx, 2] = original[idx, 2] * BASE_STRENGTH
    scaled[idx, 6] = original[idx, 6] * PARETIC_VMAX
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()

    joint_traces = {joint_key: {"l": [], "r": []} for joint_key in JOINTS}
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        for joint_key, joint_prefix in JOINTS.items():
            l_val = sim.data.joint(f"{joint_prefix}_l").qpos[0]
            r_val = sim.data.joint(f"{joint_prefix}_r").qpos[0]
            joint_traces[joint_key]["l"].append(l_val)
            joint_traces[joint_key]["r"].append(r_val)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    return joint_traces, steps


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print(f"Running stroke combo: strength={BASE_STRENGTH}, vmax={PARETIC_VMAX}, paretic_side={PARETIC_SIDE}\n")
    joint_traces, steps = run_stroke_combo(policy)
    print(f"Steps survived: {steps}\n")

    transient = min(30, steps // 4)

    print("=== Simulated stroke combo joint-angle ROM asymmetry ===")
    print(f"{'Joint':<15} {'Paretic ROM':>12} {'Nonparetic ROM':>15} {'Asymmetry':>11}")
    for joint_key in JOINTS:
        l_vals = np.degrees(np.array(joint_traces[joint_key]["l"][transient:]))
        r_vals = np.degrees(np.array(joint_traces[joint_key]["r"][transient:]))

        # paretic = left (per PARETIC_SIDE setting above)
        paretic_vals = l_vals if PARETIC_SIDE == "left" else r_vals
        nonparetic_vals = r_vals if PARETIC_SIDE == "left" else l_vals

        paretic_rom = paretic_vals.max() - paretic_vals.min()
        nonparetic_rom = nonparetic_vals.max() - nonparetic_vals.min()
        denom = paretic_rom + nonparetic_rom
        asymmetry = (nonparetic_rom - paretic_rom) / denom if denom > 0 else float('nan')

        print(f"{joint_key:<15} {paretic_rom:>12.2f} {nonparetic_rom:>15.2f} {asymmetry:>11.4f}")

    print("\n=== Real stroke patient values (for direct comparison) ===")
    print("AnkleAngles_ROM_asymmetry:  -0.0265")
    print("KneeAngles_ROM_asymmetry:   -0.1445")
    print("HipAngles_ROM_asymmetry:    -0.1321")