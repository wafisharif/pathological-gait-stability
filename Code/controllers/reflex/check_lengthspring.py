import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from myosuite.utils import gym

env = gym.make('myoLegWalk-v0')
env.reset()
sim = env.unwrapped.sim

print("tendon_lengthspring shape:", sim.model.tendon_lengthspring.shape)
print("First 5 rows:", sim.model.tendon_lengthspring[:5])
print("\nactuator_lengthrange shape:", sim.model.actuator_lengthrange.shape)
print("actuator_biasprm shape:", sim.model.actuator_biasprm.shape)

env.close()