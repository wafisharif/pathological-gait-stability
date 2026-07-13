from myosuite.utils import gym
import deprl
import numpy as np

RIGHT_LEG_IDX = slice(0, 40)
LEFT_LEG_IDX = slice(40, 80)

def run_and_check(policy, weak_side, weakness_factor=0.3, steps_to_check=20):
    """
    Use a strong, obvious weakness (0.3) so the effect on foot height
    is unmistakable, and just print feet_heights for a handful of steps.
    """
    env = gym.make('myoLegWalk-v0')
    obs, _ = env.reset()

    print(f"\n--- weak_side={weak_side}, weakness_factor={weakness_factor} ---")
    for step in range(steps_to_check):
        action = policy(obs).copy()
        if weak_side == "left":
            action[LEFT_LEG_IDX] *= weakness_factor
        elif weak_side == "right":
            action[RIGHT_LEG_IDX] *= weakness_factor

        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(env.unwrapped.sim)
        feet_heights = np.array(obs_dict["feet_heights"])
        print(f"step {step}: feet_heights = {feet_heights}")

        if terminated or truncated:
            print("  (fell/terminated)")
            break

    env.close()


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    run_and_check(policy, weak_side="left")
    run_and_check(policy, weak_side="right")