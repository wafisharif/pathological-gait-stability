"""
Custom MyoLeg walking environment combining:
1) Physiologically-grounded imitation reward (joint-angle matching against
   real motion-capture data), per Miles's approved Phase 1 plan.
2) A physiologically-grounded effort/pain cost function, using the EXACT
   formula and coefficients from Schumacher, Geijtenbeek, Caggiano, Kumar,
   Schmitt, Martius, Haeufle (2023), "Natural and Robust Walking using RL
   without Demonstrations in High-Dimensional Musculoskeletal Models"
   (arXiv:2309.02976, Table IV(d)), the direct follow-up to DEP-RL by the
   same authors who created MyoSuite, applied to this exact MyoLeg model.

Does not modify any MyoSuite files directly.

DELIBERATE ADAPTATION (documented, not a silent guess): the paper's joint-
limit pain term uses constraint torque, which requires extracting MuJoCo's
internal constraint-force arrays -- judged too risky to implement without
extensive separate verification. Replaced with a directly verifiable proxy:
penalizing joint angle proximity to its known range limit. Same intent
(discourage relying on joint limits), different, safer measurement method.

NOT YET CALIBRATED (documented, not a silent guess): the paper's adaptive
weight schedule for the effort-cubed term (alpha(t)) uses a performance
threshold (theta=1000) tuned to THEIR reward scale (r = r_vel - c_effort -
c_pain). Our reward combines several additional terms at a different scale,
so this threshold cannot be safely copied. A constant placeholder weight is
used instead; recalibrating theta based on this environment's own observed
episode-return magnitude (once real training data exists) is a specific,
well-defined follow-up task, not something to guess at now.
"""
import os
import numpy as np
import collections
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler

# ============================================================
# Imitation reward corrections (unchanged from before)
# ============================================================
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

IMITATION_RWD_WEIGHTS = {
    "hip_imitation": 3.0,
    "pelvis_imitation": 3.0,
    "knee_imitation": 0.5,
    "ankle_imitation": 0.5,
}

# ============================================================
# Effort/pain cost function -- EXACT values from Schumacher et al.
# 2023 (arXiv:2309.02976), Table IV(d)
# ============================================================
W1_SMOOTHNESS = 0.097       # action smoothing
W2_ACTIVE_MUSCLES = 1.579  # number of active muscles above threshold
ACTIVE_THRESHOLD = 0.15    # 15% activation = "active" (paper's exact value)
W3_JOINT_LIMIT = 0.131      # joint-limit pain (adapted proxy, see docstring)
W4_GRF_PAIN = 0.073         # GRF pain
GRF_BODYWEIGHT_MULTIPLIER = 1.2  # paper's exact threshold

# PLACEHOLDER -- see docstring. Not the paper's adaptive alpha(t).
EFFORT_CUBED_WEIGHT_PLACEHOLDER = 1.0

LEG_JOINTS_FOR_LIMIT_CHECK = [
    "hip_flexion_l", "hip_flexion_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "ankle_angle_r",
]

EFFORT_PAIN_RWD_WEIGHTS = {
    "effort_cubed": -EFFORT_CUBED_WEIGHT_PLACEHOLDER,
    "action_smoothness": -W1_SMOOTHNESS,
    "active_muscle_count": -W2_ACTIVE_MUSCLES,
    "joint_limit_pain": -W3_JOINT_LIMIT,
    "grf_pain": -W4_GRF_PAIN,
}


class WalkEnvV0Imitation(WalkEnvV0):

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        **WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **IMITATION_RWD_WEIGHTS,
        **EFFORT_PAIN_RWD_WEIGHTS,
        "act_mag": 0.0,  # superseded by effort_cubed; kept at 0 weight so it's
                          # still logged/comparable, but no longer double-
                          # penalizes effort alongside the new literature-exact term
    }

    def _setup(self, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        ref_data_path = os.path.join(os.path.dirname(__file__), "reference_trajectories.npz")
        ref_data = np.load(ref_data_path)
        self.corrected_curves = {}
        for joint_name, correction in JOINT_CORRECTIONS.items():
            raw_curve = ref_data[f"{joint_name}_mean"]
            self.corrected_curves[joint_name] = (raw_curve * correction["scale"]) + correction["offset"]

        self.prev_ctrl = None

        # Body weight, needed for the GRF pain term threshold -- must be set
        # BEFORE calling super()._setup(), since that call triggers an
        # internal throwaway step() which calls get_reward_dict() ->
        # _get_grf_pain() immediately, before this method would otherwise
        # continue past the super() call
        self.body_weight_n = np.sum(self.sim.model.body_mass) * 9.81

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

    def _get_effort_cubed(self):
        act = self.sim.data.act.copy()
        return np.mean(act ** 3)

    def _get_action_smoothness_and_active_count(self):
        current_ctrl = self.sim.data.ctrl.copy()
        if self.prev_ctrl is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = np.mean((current_ctrl - self.prev_ctrl) ** 2)
        self.prev_ctrl = current_ctrl

        act = self.sim.data.act.copy()
        # Normalized to a 0-1 FRACTION of muscles active, not a raw count,
        # so it's on the same per-step scale as our other reward terms
        n_active_fraction = np.sum(act > ACTIVE_THRESHOLD) / len(act)
        return smoothness_cost, n_active_fraction

    def _get_joint_limit_pain(self):
        total_violation = 0.0
        for joint_name in LEG_JOINTS_FOR_LIMIT_CHECK:
            joint_id = self.sim.model.joint(joint_name).id
            jnt_range = self.sim.model.jnt_range[joint_id]
            range_size = jnt_range[1] - jnt_range[0]
            if range_size <= 0:
                continue
            current_angle = self.sim.data.joint(joint_name).qpos[0]
            margin = 0.05 * range_size
            if current_angle < jnt_range[0] + margin:
                total_violation += (jnt_range[0] + margin - current_angle)
            elif current_angle > jnt_range[1] - margin:
                total_violation += (current_angle - (jnt_range[1] - margin))
        return total_violation

    def _get_grf_pain(self):
        r_force = self.sim.data.sensor('r_foot').data[0] + self.sim.data.sensor('r_toes').data[0]
        l_force = self.sim.data.sensor('l_foot').data[0] + self.sim.data.sensor('l_toes').data[0]
        threshold = GRF_BODYWEIGHT_MULTIPLIER * self.body_weight_n
        excess_r = max(0.0, r_force - threshold)
        excess_l = max(0.0, l_force - threshold)
        # Normalized to body weight, so this is a 0-to-~few-tenths scale
        # per step, not raw newtons -- consistent with %BW scaling already
        # used in this project's perturbation-testing work
        return (excess_r + excess_l) / self.body_weight_n

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        act_mag = (
            float(np.linalg.norm(self.obs_dict["act"], axis=-1).flatten()[0]) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        hip_imitation = self._get_imitation_reward("HipAngles")
        pelvis_imitation = self._get_imitation_reward("PelvisAngles")
        knee_imitation = self._get_imitation_reward("KneeAngles")
        ankle_imitation = self._get_imitation_reward("AnkleAngles")

        effort_cubed = self._get_effort_cubed()
        action_smoothness, active_muscle_count = self._get_action_smoothness_and_active_count()
        joint_limit_pain = self._get_joint_limit_pain()
        grf_pain = self._get_grf_pain()

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
                ("effort_cubed", effort_cubed),
                ("action_smoothness", action_smoothness),
                ("active_muscle_count", active_muscle_count),
                ("joint_limit_pain", joint_limit_pain),
                ("grf_pain", grf_pain),
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict