from myosuite.utils import gym
import deprl
import numpy as np

def record_episode(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled_gainprm = original_gainprm.copy()
    scaled_gainprm[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled_gainprm

    obs, _ = env.reset()

    feet_heights_trace = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_dict = env.unwrapped.get_obs_dict(sim)
        feet_heights_trace.append(np.array(obs_dict["feet_heights"]).copy())

        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    feet_heights_trace = np.array(feet_heights_trace)
    print(f"strength={strength_factor}: steps_survived={steps_survived}")
    print(f"  mean foot heights (R, L): {np.mean(feet_heights_trace, axis=0)}")
    return feet_heights_trace, steps_survived


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for s in [1.0, 0.8, 0.6, 0.4, 0.2]:
        record_episode(policy, strength_factor=s)