"""
Phase 1 imitation reward function: rewards the simulated walker for
matching real, phase-indexed reference joint-angle trajectories from
able-bodied motion-capture data. Combined with (not replacing) the
existing DEP-RL goal-driven reward terms already in walk_v0.py.

Per-joint corrections and confidence weights are based on direct
measurement (see compare_real_vs_sim_all_joints.py and
compare_real_vs_sim_pelvis.py):
  - Hip: shape correlates well with real data (0.862) -> HIGH weight
  - Pelvis: both signals stable, simple offset mismatch -> HIGH weight
  - Knee: weak shape correlation (0.435) -> LOW weight
  - Ankle: very weak shape correlation (0.262) -> LOW weight
This matches Miles's approved per-joint confidence-weighting decision,
not a shape/correlation-based reward (rejected as too complex).
"""
from myosuite.utils import gym
import deprl
import numpy as np
from myosuite.utils.quat_math import quat2euler

# Measured corrections: adjusted_real = (real * scale) + offset
JOINT_CORRECTIONS = {
    "HipAngles":    {"scale": 2.04, "offset": -11.50, "weight": 1.0},   # high confidence
    "PelvisAngles": {"scale": 1.0,  "offset": -93.88, "weight": 1.0},   # high confidence (offset-only)
    "KneeAngles":   {"scale": 2.40, "offset": 17.86,  "weight": 0.2},   # low confidence
    "AnkleAngles":  {"scale": 2.86, "offset": -25.71, "weight": 0.2},   # low confidence
}

MUJOCO_JOINT_NAMES = {
    "HipAngles": "hip_flexion_r",
    "KneeAngles": "knee_angle_r",
    "AnkleAngles": "ankle_angle_r",
}


def load_and_correct_reference_curves(npz_path="Results/reference_trajectories.npz"):
    """Loads real reference curves and applies the measured per-joint correction."""
    ref_data = np.load(npz_path)
    corrected = {}
    for joint_name, correction in JOINT_CORRECTIONS.items():
        raw_curve = ref_data[f"{joint_name}_mean"]
        corrected[joint_name] = (raw_curve * correction["scale"]) + correction["offset"]
    return corrected


def get_imitation_reward(sim, phase_var, corrected_curves):
    """
    Computes the combined, weighted imitation reward for one simulation
    step. Follows the same style as existing reward terms in walk_v0.py:
    np.exp(-k * diff^2), bounded between 0 and 1 per joint.
    """
    phase_index = int(np.clip(phase_var, 0.0, 1.0) * 1000)
    total_reward = 0.0
    total_weight = 0.0
    per_joint_rewards = {}

    for joint_name, correction in JOINT_CORRECTIONS.items():
        target_angle = corrected_curves[joint_name][phase_index]

        if joint_name == "PelvisAngles":
            pelvis_quat = sim.data.body('pelvis').xquat.copy()
            pelvis_euler = quat2euler(pelvis_quat)
            current_angle = np.degrees(pelvis_euler[1])
        else:
            mj_name = MUJOCO_JOINT_NAMES[joint_name]
            current_angle = np.degrees(sim.data.joint(mj_name).qpos[0])

        diff = current_angle - target_angle
        joint_reward = np.exp(-0.001 * diff**2)  # smooth, bounded 0-1

        weight = correction["weight"]
        total_reward += weight * joint_reward
        total_weight += weight
        per_joint_rewards[joint_name] = joint_reward

    combined_reward = total_reward / total_weight  # weighted average, stays in 0-1 range
    return combined_reward, per_joint_rewards


if __name__ == "__main__":
    # Quick sanity test: run the known-stable config and confirm the
    # reward behaves sensibly (bounded, varies meaningfully over time)
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    policy = deprl.load_baseline(env)

    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    scaled[:, 2] = original[:, 2] * 0.85
    sim.model.actuator_gainprm[:] = scaled

    corrected_curves = load_and_correct_reference_curves()

    obs, _ = env.reset()
    print("step, phase_var, combined_reward, hip_r, knee_r, ankle_r, pelvis_r")

    for step in range(100):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_dict = env.unwrapped.get_obs_dict(sim)
        phase = float(np.array(obs_dict["phase_var"]).flatten()[0])

        combined, per_joint = get_imitation_reward(sim, phase, corrected_curves)

        if step % 10 == 0:
            print(f"{step}, {phase:.3f}, {combined:.3f}, "
                  f"{per_joint['HipAngles']:.3f}, {per_joint['KneeAngles']:.3f}, "
                  f"{per_joint['AnkleAngles']:.3f}, {per_joint['PelvisAngles']:.3f}")

        if terminated or truncated:
            print(f"Terminated at step {step}")
            break

    env.close()