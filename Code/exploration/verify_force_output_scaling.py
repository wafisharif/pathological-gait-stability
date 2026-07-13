from myosuite.utils import gym
import deprl
import numpy as np

def get_mean_muscle_force(policy, strength_factor, max_steps=200):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled_gainprm = original_gainprm.copy()
    scaled_gainprm[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled_gainprm

    obs, _ = env.reset()
    forces = []

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(sim)
        forces.append(np.array(obs_dict["muscle_force"]).copy())
        if terminated or truncated:
            break

    env.close()
    forces = np.array(forces)
    mean_abs_force = np.mean(np.abs(forces))
    print(f"strength={strength_factor}: mean |muscle_force| across all muscles = {mean_abs_force:.2f}")


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for s in [1.0, 0.8, 0.6, 0.4, 0.2]:
        get_mean_muscle_force(policy, strength_factor=s)