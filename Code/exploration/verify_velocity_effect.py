from myosuite.utils import gym
import deprl
import numpy as np

def measure_actual_speed(policy, target_vel, max_steps=300):
    env = gym.make('myoLegWalk-v0')
    unwrapped = env.unwrapped
    unwrapped.target_x_vel = target_vel

    obs, _ = env.reset()
    sim = env.unwrapped.sim

    speeds = []
    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = unwrapped.get_obs_dict(sim)
        com_vel = np.array(obs_dict["com_vel"]).flatten()
        speeds.append(com_vel[0])  # x-direction velocity
        if terminated or truncated:
            break

    env.close()
    return np.mean(speeds), len(speeds)


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for target in [1.2, 0.8, 0.5, 0.2, 0.0]:
        mean_speed, steps = measure_actual_speed(policy, target_vel=target)
        print(f"target_x_vel={target}: actual mean x-velocity={mean_speed:.3f}, steps_survived={steps}")