from myosuite.utils import gym

env = gym.make('myoLegWalk-v0')
unwrapped = env.unwrapped

print("Current target_x_vel:", unwrapped.target_x_vel)
print("Type:", type(unwrapped.target_x_vel))

env.reset()
print("target_x_vel after reset:", unwrapped.target_x_vel)

env.close()