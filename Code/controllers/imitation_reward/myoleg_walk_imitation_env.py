"""
Custom MyoLeg walking environment that adds physiologically-grounded
imitation reward terms on top of the existing DEP-RL goal-driven reward,
per Miles's approved Phase 1 plan. Does not modify any MyoSuite files
directly -- this is a separate subclass, safe to share and version
independently of the installed package.

Per-joint confidence weighting (Miles approved):
  - Hip: high weight (shape correlates well with real data, 0.862)
  - Pelvis: high weight (stable in both real and sim, simple offset fix)
  - Knee: low weight (weak shape correlation, 0.435)
  - Ankle: low weight (very weak shape correlation, 0.262)
"""
import os
import numpy as np
import collections
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler

# Measured corrections: adjusted_real = (real * scale) + offset
JOINT_CORRECTIONS = {
    "HipAngles":    {"scale": 2.04, "offset": -11.50, "weight": 1.0},
    "PelvisAngles": {"scale": 1.0,  "offset": -93.88, "weight": 1.0},
    "KneeAngles":   {"scale": 2.40, "offset": 17.86,  "weight": 0.2},
    "AnkleAngles":  {"scale": 2.86, "offset": -25.71, "weight": 0.2},
}

MUJOCO_JOINT_NAMES = {
    "HipAngles": "hip_flexion_r",
    "KneeAngles": "knee_angle_r",
    "AnkleAngles": "ankle_angle_r",
}

# New weights added to the existing DEFAULT_RWD_KEYS_AND_WEIGHTS, not
# replacing any of them
IMITATION_RWD_WEIGHTS = {
    "hip_imitation": 3.0,
    "pelvis_imitation": 3.0,
    "knee_imitation": 0.5,
    "ankle_imitation": 0.5,
}


class WalkEnvV0Imitation(WalkEnvV0):

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        **WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **IMITATION_RWD_WEIGHTS,
    }

    def _setup(self, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        # Load the real, corrected reference curves once, at setup time
        ref_data_path = os.path.join(os.path.dirname(__file__), "reference_trajectories.npz")
        ref_data = np.load(ref_data_path)
        self.corrected_curves = {}
        for joint_name, correction in JOINT_CORRECTIONS.items():
            raw_curve = ref_data[f"{joint_name}_mean"]
            self.corrected_curves[joint_name] = (raw_curve * correction["scale"]) + correction["offset"]

        super()._setup(weighted_reward_keys=weighted_reward_keys, **kwargs)

    def _get_imitation_reward(self, joint_name):
        phase = (self.steps / self.hip_period) % 1
        phase_index = int(np.clip(phase, 0.0, 1.0) * 1000)
        target_angle = self.corrected_curves[joint_name][phase_index]

        if joint_name == "PelvisAngles":
            pelvis_quat = self.sim.data.body('pelvis').xquat.copy()
            pelvis_euler = quat2euler(pelvis_quat)
            current_angle = np.degrees(pelvis_euler[1])
        else:
            mj_name = MUJOCO_JOINT_NAMES[joint_name]
            current_angle = np.degrees(self.sim.data.joint(mj_name).qpos[0])

        diff = current_angle - target_angle
        return np.exp(-0.001 * diff**2)

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        hip_imitation = self._get_imitation_reward("HipAngles")
        pelvis_imitation = self._get_imitation_reward("PelvisAngles")
        knee_imitation = self._get_imitation_reward("KneeAngles")
        ankle_imitation = self._get_imitation_reward("AnkleAngles")

        rwd_dict = collections.OrderedDict(
            (
                ("vel_reward", vel_reward),
                ("cyclic_hip", cyclic_hip),
                ("ref_rot", ref_rot),
                ("joint_angle_rew", joint_angle_rew),
                ("act_mag", act_mag),
                ("hip_imitation", hip_imitation),
                ("pelvis_imitation", pelvis_imitation),
                ("knee_imitation", knee_imitation),
                ("ankle_imitation", ankle_imitation),
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict