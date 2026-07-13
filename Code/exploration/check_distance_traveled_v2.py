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
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    steps_survived = 0
    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    planar_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    return planar_distance, steps_survived


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    for s in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        dist, steps = measure_distance(policy, strength_factor=s)
        print(f"strength={s}: planar_distance={dist:.3f}, steps_survived={steps}, avg_speed={dist/max(steps,1):.4f}")