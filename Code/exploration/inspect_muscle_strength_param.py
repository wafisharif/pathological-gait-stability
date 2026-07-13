from myosuite.utils import gym
import numpy as np

env = gym.make('myoLegWalk-v0')
env.reset()
sim = env.unwrapped.sim

muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
gainprm = sim.model.actuator_gainprm  # shape (n_actuators, 10) typically

print("=== GAINPRM SHAPE ===")
print(gainprm.shape)

print("\n=== FIRST FEW MUSCLES: FULL GAINPRM ROW ===")
for i in range(5):
    print(muscle_names[i], "->", gainprm[i])

print("\n=== ASSUMING INDEX 2 = PEAK FORCE, ALL MUSCLES ===")
for i, name in enumerate(muscle_names):
    print(i, name, "peak_force(gainprm[2]):", gainprm[i][2])