from myosuite.utils import gym

env = gym.make('myoLegWalk-v0')
env.reset()
sim = env.unwrapped.sim

print("=== SITE NAMES (look for foot-related ones) ===")
for i in range(sim.model.nsite):
    print(i, sim.model.site(i).name)