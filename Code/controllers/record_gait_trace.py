from myosuite.utils import gym
import deprl
import numpy as np

def record_episode(policy, weakness_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    trace = {
        "feet_heights": [],
        "feet_rel_positions": [],
        "qpos_without_xy": [],
        "phase_var": [],
    }

    steps_survived = 0
    total_reward = 0

    for step in range(max_steps):
        action = policy(obs)
        weakened_action = action * weakness_factor
        obs, reward, terminated, truncated, info = env.step(weakened_action)

        # Pull the named observation dict (same method as our first
        # inspection script) so we record interpretable quantities,
        # not just the raw 403-length obs vector.
        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)

        trace["feet_heights"].append(np.array(obs_dict["feet_heights"]).copy())
        trace["feet_rel_positions"].append(np.array(obs_dict["feet_rel_positions"]).copy())
        trace["qpos_without_xy"].append(np.array(obs_dict["qpos_without_xy"]).copy())
        trace["phase_var"].append(np.array(obs_dict["phase_var"]).copy())

        total_reward += reward
        steps_survived += 1

        if terminated or truncated:
            break

    env.close()

    # Convert lists of per-step arrays into single stacked arrays
    for key in trace:
        trace[key] = np.stack(trace[key])

    print(f"weakness={weakness_factor}: steps_survived={steps_survived}, total_reward={total_reward:.2f}")
    return trace, steps_survived, total_reward


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    weakness_levels_to_record = [1.0, 0.8, 0.6, 0.4, 0.2]

    for w in weakness_levels_to_record:
        trace, steps, reward = record_episode(policy, weakness_factor=w)
        filename = f"Code/controllers/gait_trace_w{w}.npy"
        np.save(filename, trace, allow_pickle=True)
        print(f"  saved to {filename}")