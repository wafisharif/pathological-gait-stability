from myosuite.utils import gym
import inspect

env = gym.make('myoLegWalk-v0')
unwrapped = env.unwrapped

print("=== Trying to set target_x_vel directly ===")
unwrapped.target_x_vel = 0.5
print("After manual set:", unwrapped.target_x_vel)

print("\n=== Checking reset() signature for velocity-related args ===")
print(inspect.signature(unwrapped.reset))

print("\n=== Searching env source for 'target_x_vel' usage ===")
source_file = inspect.getfile(unwrapped.__class__)
print("Defined in:", source_file)

with open(source_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'target_x_vel' in line:
            print(f"  line {i}: {line.strip()}")

env.close()