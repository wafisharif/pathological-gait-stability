from myosuite.utils import gym
import deprl
import numpy as np

def measure_distance(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    # Track actual root/pelvis x-position directly, not velocity
    start_x = sim.data.qpos[0]  # x is typically the first qpos entry (global x position)

    steps_survived = 0
    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_survived += 1
        if terminated or truncated:
            break

    end_x = sim.data.qpos[0]
    env.close()

    distance = end_x - start_x
    return distance, steps_survived


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for s in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        dist, steps = measure_distance(policy, strength_factor=s)
        print(f"strength={s}: distance_traveled={dist:.3f}, steps_survived={steps}")