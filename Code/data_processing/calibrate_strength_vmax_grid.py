from myosuite.utils import gym
import deprl
import numpy as np

REAL_HEALTHY_DS_PCT = 28.225250
REAL_ALS_DS_PCT = 40.766620


def get_ground_arrays(heights, foot_index):
    h = heights[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return np.zeros(len(h), dtype=bool)
    threshold = np.min(h) + 0.25 * height_range
    return h < threshold


def measure_double_support(policy, strength_factor, vmax_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    scaled[:, 6] = original_gainprm[:, 6] * vmax_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    feet_heights = []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(sim)
        feet_heights.append(np.array(obs_dict["feet_heights"]).copy())
        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    feet_heights = np.array(feet_heights)
    transient = min(30, steps_survived // 4)
    feet_post = feet_heights[transient:]

    if len(feet_post) < 5:
        return float('nan'), steps_survived, 0

    on_ground_r = get_ground_arrays(feet_post, foot_index=0)
    on_ground_l = get_ground_arrays(feet_post, foot_index=1)
    both_on = on_ground_r & on_ground_l
    n_strikes = len([i for i in range(1, len(on_ground_r)) if on_ground_r[i] and not on_ground_r[i-1]])

    ds_pct = np.sum(both_on) / len(on_ground_r) * 100
    return ds_pct, steps_survived, n_strikes


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print(f"Target: Healthy DS%={REAL_HEALTHY_DS_PCT:.1f}, ALS DS%={REAL_ALS_DS_PCT:.1f}\n")

    # Grid: combine strength reduction with vmax reduction at each level,
    # since vmax already proved it extends the graceful survival range
    strength_levels = [0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    vmax_levels = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]

    print("strength, vmax, ds_pct, steps_survived, n_strikes, gap_to_ALS")
    for s in strength_levels:
        for v in vmax_levels:
            ds_pct, steps, n_strikes = measure_double_support(policy, strength_factor=s, vmax_factor=v)
            gap = REAL_ALS_DS_PCT - ds_pct if not np.isnan(ds_pct) else float('nan')
            confidence_flag = "ok" if n_strikes >= 6 else "low_conf"
            print(f"{s}, {v}, {ds_pct:.2f}, {steps}, {n_strikes} ({confidence_flag}), {gap:.2f}")