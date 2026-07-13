from myosuite.utils import gym
import numpy as np

env = gym.make('myoLegWalk-v0')
env.reset()
sim = env.unwrapped.sim

muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
dynprm = sim.model.actuator_dynprm

print("=== DYNPRM SHAPE ===")
print(dynprm.shape)

print("\n=== FIRST FEW MUSCLES: FULL DYNPRM ROW ===")
for i in range(5):
    print(muscle_names[i], "->", dynprm[i])

print("\n=== tau_act / tau_deact (commonly indices 0 and 1) ACROSS ALL MUSCLES ===")
for i, name in enumerate(muscle_names):
    print(i, name, "dynprm[0]:", dynprm[i][0], "dynprm[1]:", dynprm[i][1])

env.close()