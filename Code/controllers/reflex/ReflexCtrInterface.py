# Author(s): Seungmoon Song <seungmoon.song@gmail.com>, Chun Kwang Tan <riodren.tan@gmail.com>
"""
adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.

NOTE: This file has been patched by Wafi (Pathological Gait Stability project,
June 2026) to work with a current MuJoCo/MyoSuite installation. The original
tutorial file (from MyoHub/myosuite) used the older mujoco_py-style API
(get_body_xpos, get_sensor, get_joint_qpos, actuator_names.index, etc.),
which no longer exists in current MuJoCo Python bindings. All such calls
below have been translated to the modern named-access API
(model.body(name), data.sensor(name).data, model.joint(name).qposadr, etc.).

Known remaining uncertainty (flagged, not silently guessed):
- The original code read `get_body_xquat('root')` in run_reflex_step(). This
  model does not appear to have a body literally named 'root' (the free
  root joint's body is 'pelvis'). Changed to 'pelvis' below -- VERIFY this
  is correct by checking termination behavior makes sense once running.
"""

from __future__ import division  # '/' always means non-truncating division
import numpy as np
import mujoco
from reflexCtr import LocoCtrl

import myosuite
from myosuite.utils import gym  # FIXED: use gymnasium-based gym, matches register_env_variant's registry

import os
from scipy.spatial.transform import Rotation as R

from myosuite.envs.env_variants import register_env_variant
from myosuite.utils.quat_math import quat2euler
from myosuite.utils.quat_math import euler2quat


class MyoLegReflex(object):

    DEFAULT_INIT_POSE = {}
    DEFAULT_INIT_POSE['model_pose'] = {'yaw': np.deg2rad(0), 'pitch': np.deg2rad(15), 'roll': np.deg2rad(0)}
    DEFAULT_INIT_POSE['model_height'] = 0.92
    DEFAULT_INIT_POSE['joint_angles'] = {
        'hip_flexion_r': np.deg2rad(180 - 190),
        'hip_flexion_l': np.deg2rad(180 - 155),
        'knee_angle_r': np.deg2rad(180 - 165),
        'knee_angle_l': np.deg2rad(180 - 180),
        'ankle_angle_r': np.deg2rad(90 - 90),
        'ankle_angle_l': np.deg2rad(90 - 100),
    }
    DEFAULT_INIT_POSE['velocity'] = {'cartesian': [1.5, 0.0, 0.0]}

    def __init__(self, init_dict=DEFAULT_INIT_POSE, dt=0.01, mode='3D', sim_time=2.0, seed=0):
        self.dt = dt
        self.t = 0
        self.mode = mode

        self.n_par = len(LocoCtrl.cp_keys)
        control_dimension = 3
        self.cp_map = LocoCtrl.cp_map
        self.ReflexCtrl = LocoCtrl(self.dt, control_dimension=control_dimension, params=np.ones(self.n_par))

        self.sim_time = sim_time
        self.timestep_limit = int(self.sim_time / self.dt)

        self.init_dict = init_dict
        self.seed = seed

        # FIXED: absolute path to the real installed myolegs.xml (was a broken
        # relative path assuming a specific working directory)
        myolegs_xml_path = '/opt/anaconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml'

        # FIXED: env_id changed from non-existent 'myoLegDemo-v0' to
        # 'myoLegStandRandom-v0', per the official MyoSuite fix (PR #233)
        register_env_variant(
            env_id='myoLegStandRandom-v0',
            variants={'model_path': myolegs_xml_path,
                      'normalize_act': False},
            variant_id='MyoLegReflex-v0',
            silent=False
        )
        self.env = gym.make('MyoLegReflex-v0')

        print(f"Seed added - ", seed)
        # REMOVED: print('List of cameras available', self.env.sim.model.camera_names)
        # -- camera_names no longer exists on MjModel in current bindings; this
        # was purely informational/diagnostic, not needed for control.
        self.env.reset()
        self.env.unwrapped.seed(seed)

        self.muscle_labels = {}
        self.muscles_dict = {}
        self.muscle_Fmax = {}
        self.muscle_L0 = {}

        self.init_pelvis = np.zeros(3, )

        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 0
        self.footstep['l_contact'] = 0

        self.cp = self.ReflexCtrl.cp

    # -----------------------------------------------------------------------------------------------------------------
    def _actuator_index(self, name):
        # FIXED: replaces self.env.sim.model.actuator_names.index(name), which
        # no longer exists. Builds the name list once, then looks up by name.
        if not hasattr(self, '_actuator_name_list'):
            self._actuator_name_list = [self.env.sim.model.actuator(i).name for i in range(self.env.sim.model.nu)]
        return self._actuator_name_list.index(name)

    def _body_vel(self, body_name):
        body_id = self.env.sim.model.body(body_name).id
        m = self.env.sim.model.ptr  # unwrap dm_control wrapper to raw MjModel
        d = self.env.sim.data.ptr   # unwrap dm_control wrapper to raw MjData
        res = np.zeros((6, 1))
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 0)
        res = res.flatten()
        ang_vel = res[0:3].copy()
        lin_vel = res[3:6].copy()
        return lin_vel, ang_vel

    # -----------------------------------------------------------------------------------------------------------------
    def reset(self):
        self.env.reset()
        self.env.unwrapped.seed(self.seed)

        self.ReflexCtrl.reset()

        self._set_muscle_groups()
        self._set_initial_pose(self.init_dict)

    # -----------------------------------------------------------------------------------------------------------------
    def update(self):
        self.t += self.dt
        self.ReflexCtrl.update(self.get_obs_dict())
        return self.ReflexCtrl.stim.copy()

    # -----------------------------------------------------------------------------------------------------------------
    def set_control_params(self, params):
        self.ReflexCtrl.set_control_params(params)

    # -----------------------------------------------------------------------------------------------------------------
    def set_control_params_RL(self, s_leg, params):
        self.ReflexCtrl.set_control_params_RL(s_leg, params)

    # -----------------------------------------------------------------------------------------------------------------
    def get_obs_dict(self):
        # FIXED: all get_body_x*/get_sensor/get_joint_q* calls translated to
        # modern named-access API (model/data .body()/.sensor()/.joint()).

        pel_quat = self.env.sim.data.body('pelvis').xquat.copy()
        pel_euler = quat2euler(pel_quat)
        pelvis_roll = pel_euler[0] - (np.pi / 2)
        pelvis_pitch = pel_euler[2] * (-1)
        pelvis_yaw = pel_euler[1] * (-1)

        lin_vel, ang_vel = self._body_vel('pelvis')
        temp_seg_vel = lin_vel
        dx_local, dy_local = self.rotate_frame(temp_seg_vel[0], temp_seg_vel[1], pelvis_yaw)
        pelvis_vel = np.hstack((np.array([dx_local, dy_local, -1 * temp_seg_vel[2]]), ang_vel))

        temp_right = (self.env.sim.data.sensor('r_foot').data[0] + self.env.sim.data.sensor('r_toes').data[0])
        temp_left = (self.env.sim.data.sensor('l_foot').data[0] + self.env.sim.data.sensor('l_toes').data[0])

        sensor_data = {'body': {}, 'r_leg': {}, 'l_leg': {}}
        sensor_data['body']['theta'] = [pelvis_roll, pelvis_pitch]
        sensor_data['body']['d_pos'] = [pelvis_vel[0], pelvis_vel[1]]
        sensor_data['body']['dtheta'] = [pelvis_vel[3], pelvis_vel[4]]

        sensor_data['r_leg']['load_ipsi'] = temp_right / (np.sum(self.env.sim.model.body_mass) * 9.8)
        sensor_data['l_leg']['load_ipsi'] = temp_left / (np.sum(self.env.sim.model.body_mass) * 9.8)

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            sensor_data[s_leg]['contact_ipsi'] = 1 if sensor_data[s_leg]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['contact_contra'] = 1 if sensor_data[s_legc]['load_ipsi'] > 0.1 else 0
            sensor_data[s_leg]['load_contra'] = sensor_data[s_legc]['load_ipsi']

            sensor_data[s_leg]['phi_hip'] = (np.pi - self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0])
            sensor_data[s_leg]['phi_knee'] = (np.pi - self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0])
            sensor_data[s_leg]['phi_ankle'] = (0.5 * np.pi - self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0])
            sensor_data[s_leg]['dphi_knee'] = -1 * self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel[0]

            sensor_data[s_leg]['alpha'] = sensor_data[s_leg]['phi_hip'] - 0.5 * sensor_data[s_leg]['phi_knee']
            dphi_hip = -1 * self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel[0]
            sensor_data[s_leg]['dalpha'] = dphi_hip - 0.5 * sensor_data[s_leg]['dphi_knee']

            sensor_data[s_leg]['alpha_f'] = -1 * (-1 * self.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0]) + 0.5 * np.pi

            temp_mus_force = self.env.sim.data.actuator_force.copy()

            sensor_data[s_leg]['F_RF'] = -1 * np.mean(temp_mus_force[self.muscles_dict[s_leg]['RF']] / (self.muscle_Fmax[s_leg]['RF']))
            sensor_data[s_leg]['F_VAS'] = -1 * np.mean(temp_mus_force[self.muscles_dict[s_leg]['VAS']] / (self.muscle_Fmax[s_leg]['VAS']))
            sensor_data[s_leg]['F_GAS'] = -1 * np.mean(temp_mus_force[self.muscles_dict[s_leg]['GAS']] / (self.muscle_Fmax[s_leg]['GAS']))
            sensor_data[s_leg]['F_SOL'] = -1 * np.mean(temp_mus_force[self.muscles_dict[s_leg]['SOL']] / (self.muscle_Fmax[s_leg]['SOL']))

        return sensor_data

    # ---------------------------------------------------------------------------------------------------
    def run_reflex_step(self):
        is_done = False

        new_act = self.reflex2mujoco(self.update())
        self.env.step(new_act)

        self.update_footstep()

        out_dict = self.get_obs_dict()

        # FLAGGED CHANGE: original used get_body_xquat('root'); using 'pelvis'
        # here since no separate 'root' body was found. Verify is_done
        # triggers sensibly (i.e. when the model visibly falls) once running.
        temp_pel_euler = quat2euler(self.env.sim.data.body('pelvis').xquat.copy())

        if self.env.sim.data.body('pelvis').xpos[2] < 0.65:
            is_done = True
        if temp_pel_euler[1] < np.deg2rad(-30) or temp_pel_euler[1] > np.deg2rad(30):
            is_done = True

        return [out_dict, is_done, np.round(self.env.sim.data.time, 2), new_act]

    # ---------- Initialization Functions ----------
    def _set_muscle_groups(self):
        glu_r = [self._actuator_index('glmax1_r'), self._actuator_index('glmax2_r'),
                  self._actuator_index('glmax3_r'), self._actuator_index('glmed3_r')]
        glu_l = [self._actuator_index('glmax1_l'), self._actuator_index('glmax2_l'),
                  self._actuator_index('glmax3_l'), self._actuator_index('glmed3_l')]
        glu_r_lbl = ['glmax1_r', 'glmax2_r', 'glmax3_r', 'glmed3_r']
        glu_l_lbl = ['glmax1_l', 'glmax2_l', 'glmax3_l', 'glmed3_l']

        ham_r = [self._actuator_index('semimem_r'), self._actuator_index('semiten_r'), self._actuator_index('bflh_r')]
        ham_l = [self._actuator_index('semimem_l'), self._actuator_index('semiten_l'), self._actuator_index('bflh_l')]
        ham_r_lbl = ['semimem_r', 'semiten_r', 'bflh_r']
        ham_l_lbl = ['semimem_l', 'semiten_l', 'bflh_l']

        bfsh_r = [self._actuator_index('bfsh_r')]
        bfsh_l = [self._actuator_index('bfsh_l')]
        bfsh_r_lbl = ['bfsh_r']
        bfsh_l_lbl = ['bfsh_l']

        gas_r = [self._actuator_index('gaslat_r'), self._actuator_index('gasmed_r')]
        gas_l = [self._actuator_index('gaslat_l'), self._actuator_index('gasmed_l')]
        gas_r_lbl = ['gaslat_r', 'gasmed_r']
        gas_l_lbl = ['gaslat_l', 'gasmed_l']

        sol_r = [self._actuator_index('soleus_r'), self._actuator_index('perbrev_r'),
                  self._actuator_index('perlong_r'), self._actuator_index('tibpost_r')]
        sol_l = [self._actuator_index('soleus_l'), self._actuator_index('perbrev_l'),
                  self._actuator_index('perlong_l'), self._actuator_index('tibpost_l')]
        sol_r_lbl = ['soleus_r', 'perbrev_r', 'perlong_r', 'tibpost_r']
        sol_l_lbl = ['soleus_l', 'perbrev_l', 'perlong_l', 'tibpost_l']

        hfl_r = [self._actuator_index('psoas_r'), self._actuator_index('iliacus_r')]
        hfl_l = [self._actuator_index('psoas_l'), self._actuator_index('iliacus_l')]
        hfl_r_lbl = ['psoas_r', 'iliacus_r']
        hfl_l_lbl = ['psoas_l', 'iliacus_l']

        hab_r = [self._actuator_index('piri_r'), self._actuator_index('sart_r'),
                  self._actuator_index('glmed1_r'), self._actuator_index('glmed2_r'),
                  self._actuator_index('glmin1_r'), self._actuator_index('glmin2_r'),
                  self._actuator_index('glmin3_r')]
        hab_l = [self._actuator_index('piri_l'), self._actuator_index('sart_l'),
                  self._actuator_index('glmed1_l'), self._actuator_index('glmed2_l'),
                  self._actuator_index('glmin1_l'), self._actuator_index('glmin2_l'),
                  self._actuator_index('glmin3_l')]
        hab_r_lbl = ['piri_r', 'sart_r', 'glmed1_r', 'glmed2_r', 'glmin1_r', 'glmin2_r', 'glmin3_r']
        hab_l_lbl = ['piri_l', 'sart_l', 'glmed1_l', 'glmed2_l', 'glmin1_l', 'glmin2_l', 'glmin3_l']

        had_r = [self._actuator_index('addbrev_r'), self._actuator_index('addlong_r'),
                  self._actuator_index('addmagDist_r'), self._actuator_index('addmagIsch_r'),
                  self._actuator_index('addmagMid_r'), self._actuator_index('addmagProx_r'),
                  self._actuator_index('grac_r')]
        had_l = [self._actuator_index('addbrev_l'), self._actuator_index('addlong_l'),
                  self._actuator_index('addmagDist_l'), self._actuator_index('addmagIsch_l'),
                  self._actuator_index('addmagMid_l'), self._actuator_index('addmagProx_l'),
                  self._actuator_index('grac_l')]
        had_r_lbl = ['addbrev_r', 'addlong_r', 'addmagDist_r', 'addmagIsch_r', 'addmagMid_r', 'addmagProx_r', 'grac_r']
        had_l_lbl = ['addbrev_l', 'addlong_l', 'addmagDist_l', 'addmagIsch_l', 'addmagMid_l', 'addmagProx_l', 'grac_l']

        rf_r = [self._actuator_index('recfem_r')]
        rf_l = [self._actuator_index('recfem_l')]
        rf_r_lbl = ['recfem_r']
        rf_l_lbl = ['recfem_l']

        vas_r = [self._actuator_index('vasint_r'), self._actuator_index('vaslat_r'), self._actuator_index('vasmed_r')]
        vas_l = [self._actuator_index('vasint_l'), self._actuator_index('vaslat_l'), self._actuator_index('vasmed_l')]
        vas_r_lbl = ['vasint_r', 'vaslat_r', 'vasmed_r']
        vas_l_lbl = ['vasint_l', 'vaslat_l', 'vasmed_l']

        ta_r = [self._actuator_index('tibant_r')]
        ta_l = [self._actuator_index('tibant_l')]
        ta_r_lbl = ['tibant_r']
        ta_l_lbl = ['tibant_l']

        self.muscles_dict['r_leg'] = {'HAB': hab_r, 'HAD': had_r, 'GLU': glu_r, 'HAM': ham_r, 'BFSH': bfsh_r,
                                        'GAS': gas_r, 'SOL': sol_r, 'HFL': hfl_r, 'RF': rf_r, 'VAS': vas_r, 'TA': ta_r}
        self.muscles_dict['l_leg'] = {'HAB': hab_l, 'HAD': had_l, 'GLU': glu_l, 'HAM': ham_l, 'BFSH': bfsh_l,
                                        'GAS': gas_l, 'SOL': sol_l, 'HFL': hfl_l, 'RF': rf_l, 'VAS': vas_l, 'TA': ta_l}

        self.muscle_labels['r_leg'] = {'HAB': hab_r_lbl, 'HAD': had_r_lbl, 'GLU': glu_r_lbl, 'HAM': ham_r_lbl,
                                         'BFSH': bfsh_r_lbl, 'GAS': gas_r_lbl, 'SOL': sol_r_lbl, 'HFL': hfl_r_lbl,
                                         'RF': rf_r_lbl, 'VAS': vas_r_lbl, 'TA': ta_r_lbl}
        self.muscle_labels['l_leg'] = {'HAB': hab_l_lbl, 'HAD': had_l_lbl, 'GLU': glu_l_lbl, 'HAM': ham_l_lbl,
                                         'BFSH': bfsh_l_lbl, 'GAS': gas_l_lbl, 'SOL': sol_l_lbl, 'HFL': hfl_l_lbl,
                                         'RF': rf_l_lbl, 'VAS': vas_l_lbl, 'TA': ta_l_lbl}

        # FIXED: tendon_lengthspring is now shape (n,2) [min,max] instead of
        # (n,) -- both columns were confirmed identical, so [:,0] is safe.
        temp_L0 = (self.env.sim.model.actuator_lengthrange[:, 0] - self.env.sim.model.tendon_lengthspring[:, 0]) / self.env.sim.model.actuator_biasprm[:, 0]

        for x in self.muscles_dict:
            self.muscle_Fmax[x] = {}
            self.muscle_L0[x] = {}
            for y in self.muscles_dict[x]:
                self.muscle_Fmax[x][y] = self.env.sim.model.actuator_biasprm[self.muscles_dict[x][y], 2].copy()
                self.muscle_L0[x][y] = temp_L0[self.muscles_dict[x][y]]

    def _set_initial_pose(self, init_dict):
        # FIXED: get_body_xpos -> body().xpos
        self.init_pelvis = self.env.sim.data.body('pelvis').xpos.copy()

        temp_quat_util = euler2quat([init_dict['model_pose']['roll'],
                                       init_dict['model_pose']['pitch'],
                                       init_dict['model_pose']['yaw']])

        self.env.sim.data.qpos[3] = temp_quat_util[0]
        self.env.sim.data.qpos[4] = temp_quat_util[1]
        self.env.sim.data.qpos[5] = temp_quat_util[2]
        self.env.sim.data.qpos[6] = temp_quat_util[3]

        self.env.sim.data.qvel[0] = init_dict['velocity']['cartesian'][0]
        self.env.sim.data.qvel[1] = init_dict['velocity']['cartesian'][1]
        self.env.sim.data.qvel[2] = init_dict['velocity']['cartesian'][2]

        # FIXED: original used joint_names.index(i) + a manual +7 offset,
        # assuming joint ordering lines up with qpos layout after the free
        # joint. qposadr is the robust, direct way to get the real qpos
        # index for a named joint -- no offset arithmetic/ordering assumption needed.
        for i in init_dict['joint_angles'].keys():
            qpos_addr = self.env.sim.model.joint(i).qposadr[0]
            self.env.sim.data.qpos[qpos_addr] = init_dict['joint_angles'][i]

        if 'height_offset' in init_dict.keys():
            height_offset = init_dict['height_offset']
        else:
            height_offset = 0

        self.env.sim.data.qpos[0] = 0
        self.env.sim.data.qpos[1] = 0
        self.env.sim.data.qpos[2] = init_dict['model_height'] + height_offset

        self.env.sim.forward()

    # ---------- Internal functions ----------
    def update_footstep(self):
        # FIXED: get_sensor -> sensor().data[0]
        r_contact = True if (self.env.sim.data.sensor('r_foot').data[0]) > 0.1 * (np.sum(self.env.sim.model.body_mass) * 9.8) else False
        l_contact = True if (self.env.sim.data.sensor('l_foot').data[0]) > 0.1 * (np.sum(self.env.sim.model.body_mass) * 9.8) else False

        self.footstep['new'] = False
        if ((not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact)):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def reflex2mujoco(self, output):
        mus_act = np.zeros((80,))
        mus_act[:] = 0

        legs = ['r_leg', 'l_leg']
        musc_idx = self.muscles_dict['r_leg'].keys()

        for s_leg in legs:
            for musc in musc_idx:
                mus_act[self.muscles_dict[s_leg][musc]] = output[s_leg][musc]

        return mus_act

    def rotate_frame(self, x, y, theta):
        x_rot = np.cos(theta) * x - np.sin(theta) * y
        y_rot = np.sin(theta) * x + np.cos(theta) * y
        return x_rot, y_rot