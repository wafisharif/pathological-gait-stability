import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ReflexCtrInterface import MyoLegReflex
import numpy as np

myo_env = MyoLegReflex()
myo_env.reset()
real_params = np.loadtxt('params_3D_init.txt')
print(f"Loaded params shape: {real_params.shape}, expected n_par: {myo_env.n_par}")
myo_env.set_control_params(real_params)

start_x = myo_env.env.sim.data.body('pelvis').xpos[0]
start_y = myo_env.env.sim.data.body('pelvis').xpos[1]

print(f"Starting pelvis pos: x={start_x:.4f}, y={start_y:.4f}")

for i in range(200):
    out_dict, is_done, sim_time, action = myo_env.run_reflex_step()
    if i % 20 == 0:
        x = myo_env.env.sim.data.body('pelvis').xpos[0]
        y = myo_env.env.sim.data.body('pelvis').xpos[1]
        z = myo_env.env.sim.data.body('pelvis').xpos[2]
        print(f"step {i}: x={x:.4f}, y={y:.4f}, z={z:.4f}, is_done={is_done}")
    if is_done:
        print(f"Fell at step {i}")
        break

print("Done.")