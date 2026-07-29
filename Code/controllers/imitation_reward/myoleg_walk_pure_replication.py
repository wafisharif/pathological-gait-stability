import os
import numpy as np
import collections
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0

# Effort/pain cost function
W1_SMOOTHNESS = 0.097
W2_ACTIVE_MUSCLES = 1.579
ACTIVE_THRESHOLD = 0.15
W3_JOINT_LIMIT = 0.131
W4_GRF_PAIN = 0.073
GRF_BODYWEIGHT_MULTIPLIER = 1.2

LEG_JOINTS_FOR_LIMIT_CHECK = [
    "hip_flexion_l", "hip_flexion_r",
    "knee_angle_l", "knee_angle_r",
    "ankle_angle_l", "ankle_angle_r",
]

# Adaptive effort-weight schedule
DELTA_ALPHA_INIT = 9e-4
THETA = 1000
BETA = 0.8
LAMBDA_DECAY = 0.9

TARGET_FORWARD_VEL = 1.2


class WalkEnvV0PureReplication(WalkEnvV0):

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 1.0,
        "done": 0.0,          # not part of paper's formula
        "cyclic_hip": 0.0,     # not part of paper's formula
        "ref_rot": 0.0,        # not part of paper's formula
        "joint_angle_rew": 0.0,  # not part of paper's formula
        "act_mag": 0.0,
        "effort_cubed": -1.0,   # placeholder
        "action_smoothness": -W1_SMOOTHNESS,
        "active_muscle_count": -W2_ACTIVE_MUSCLES,
        "joint_limit_pain": -W3_JOINT_LIMIT,
        "grf_pain": -W4_GRF_PAIN,
    }

    def _setup(self, weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS, **kwargs):
        self.prev_ctrl = None

        self.alpha_t = 0.0
        self.r_mean = 0.0
        self.s_mean = 0.0
        self.delta_alpha = DELTA_ALPHA_INIT
        self.current_episode_return = 0.0

        self.body_weight_n = np.sum(self.sim.model.body_mass) * 9.81 if hasattr(self, 'sim') else None

        super()._setup(weighted_reward_keys=weighted_reward_keys, **kwargs)

        self.body_weight_n = np.sum(self.sim.model.body_mass) * 9.81

    def _get_paper_vel_reward(self):
        com_vel = self._get_com_velocity()
        v = com_vel[1]
        if v < TARGET_FORWARD_VEL:
            return float(np.exp(-np.square(TARGET_FORWARD_VEL - v)))
        else:
            return 1.0

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
        return (excess_r + excess_l) / self.body_weight_n

    def _update_adaptive_alpha(self, episode_return):
        self.r_mean = BETA * self.r_mean + (1 - BETA) * episode_return

        performance_high = self.r_mean > THETA
        long_enough = self.s_mean > 0.5

        if performance_high and not long_enough:
            self.delta_alpha = LAMBDA_DECAY * self.delta_alpha
        elif performance_high and long_enough:
            self.alpha_t = self.alpha_t + self.delta_alpha
        else:
            self.alpha_t = self.alpha_t - self.delta_alpha

        self.alpha_t = max(0.0, self.alpha_t)

        c_target = 1.0 if performance_high else 0.0
        self.s_mean = BETA * self.s_mean + (1 - BETA) * c_target

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_paper_vel_reward()
        effort_cubed = self._get_effort_cubed()
        action_smoothness, active_muscle_count = self._get_action_smoothness_and_active_count()
        joint_limit_pain = self._get_joint_limit_pain()
        grf_pain = self._get_grf_pain()

        rwd_dict = collections.OrderedDict(
            (
                ("vel_reward", vel_reward),
                ("done", self._get_done()),
                ("cyclic_hip", 0.0),
                ("ref_rot", 0.0),
                ("joint_angle_rew", 0.0),
                ("act_mag", 0.0),
                ("effort_cubed", effort_cubed),
                ("action_smoothness", action_smoothness),
                ("active_muscle_count", active_muscle_count),
                ("joint_limit_pain", joint_limit_pain),
                ("grf_pain", grf_pain),
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done_flag", self._get_done()),
            )
        )

        weights = dict(self.rwd_keys_wt)
        weights["effort_cubed"] = -self.alpha_t

        rwd_dict["dense"] = np.sum(
            [weights[key] * rwd_dict[key] for key in weights.keys() if key in rwd_dict], axis=0
        )

        self.current_episode_return += rwd_dict["dense"]
        if rwd_dict["done_flag"]:
            self._update_adaptive_alpha(self.current_episode_return)
            self.current_episode_return = 0.0

        return rwd_dict