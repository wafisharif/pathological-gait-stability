from myosuite.utils import gym
import deprl
import numpy as np

def measure_speed_and_survival(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    speeds = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(sim)
        com_vel = np.array(obs_dict["com_vel"]).flatten()
        speeds.append(com_vel[0])
        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    return np.mean(speeds), steps_survived


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for s in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        mean_speed, steps = measure_speed_and_survival(policy, strength_factor=s)
        print(f"strength={s}: mean_x_speed={mean_speed:.3f}, steps_survived={steps}")