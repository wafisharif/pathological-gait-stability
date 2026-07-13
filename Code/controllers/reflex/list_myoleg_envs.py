from myosuite.utils import gym

all_envs = gym.envs.registry.keys()
myoleg_envs = [e for e in all_envs if 'myoLeg' in e or 'MyoLeg' in e]

print("=== All registered MyoLeg-related environments ===")
for e in sorted(myoleg_envs):
    print(e)