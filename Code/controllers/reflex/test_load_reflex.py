import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # so 'reflexCtr' and 'ReflexCtrInterface' import correctly

from ReflexCtrInterface import MyoLegReflex
import numpy as np

print("Creating MyoLegReflex...")
myo_env = MyoLegReflex()
print("Created successfully. Resetting...")
myo_env.reset()
print("Reset successful.")

print("Setting default control params (all ones)...")
n_par = myo_env.n_par
myo_env.set_control_params(np.ones(n_par))
print(f"n_par = {n_par}")

print("Running 10 reflex steps...")
for i in range(10):
    out_dict, is_done, sim_time, action = myo_env.run_reflex_step()
    print(f"step {i}: sim_time={sim_time}, is_done={is_done}")
    if is_done:
        print("Fell/terminated early.")
        break

print("Test complete.")