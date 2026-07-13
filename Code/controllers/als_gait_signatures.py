from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

TRANSIENT_STEPS = 30  # skip initial transient, consistent with earlier validation


def get_strikes(heights, foot_index):
    h = heights[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return []
    threshold = np.min(h) + 0.25 * height_range
    on_ground = h < threshold
    strikes = [i for i in range(1, len(on_ground)) if on_ground[i] and not on_ground[i - 1]]
    return strikes, on_ground


def run_and_measure(policy, strength_factor, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

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

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    feet_heights = np.array(feet_heights)
    if feet_heights.shape[0] < TRANSIENT_STEPS + 10:
        return None  # too short to measure meaningfully

    # Exclude transient
    feet_heights_post = feet_heights[TRANSIENT_STEPS:]

    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    strikes_r, on_ground_r = get_strikes(feet_heights_post, foot_index=0)
    strikes_l, on_ground_l = get_strikes(feet_heights_post, foot_index=1)

    n_strides = len(strikes_r) + len(strikes_l)
    cadence_steps_per_1000 = n_strides  # raw count; convert to per-1000-steps for comparability across episodes of different length
    cadence_normalized = n_strides / len(feet_heights_post) * 1000  # strides per 1000 sim steps

    stride_len = distance / max(n_strides, 1)  # rough: total distance / total strides

    def cv(strike_list):
        if len(strike_list) < 2:
            return float('nan')
        durations = np.diff(strike_list)
        return np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else float('nan')

    cv_r = cv(strikes_r)
    cv_l = cv(strikes_l)

    # Double support: both feet "on ground" simultaneously; single support: exactly one
    both_on = on_ground_r & on_ground_l
    only_r = on_ground_r & ~on_ground_l
    only_l = on_ground_l & ~on_ground_r
    n = len(on_ground_r)

    double_support_pct = np.sum(both_on) / n * 100
    single_support_pct = (np.sum(only_r) + np.sum(only_l)) / n * 100

    return {
        "strength_factor": strength_factor,
        "steps_survived": steps_survived,
        "distance": distance,
        "cadence_per_1000steps": cadence_normalized,
        "stride_length_approx": stride_len,
        "cv_stride_duration_R": cv_r,
        "cv_stride_duration_L": cv_l,
        "double_support_pct": double_support_pct,
        "single_support_pct": single_support_pct,
    }


def log_result(filepath, row, header):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    strength_levels = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    results_path = "Results/als_gait_signatures.csv"
    header = ["strength_factor", "steps_survived", "distance", "cadence_per_1000steps",
               "stride_length_approx", "cv_stride_duration_R", "cv_stride_duration_L",
               "double_support_pct", "single_support_pct"]

    print("strength, steps, distance, cadence/1000, stride_len, cv_R, cv_L, double_support%, single_support%")
    for s in strength_levels:
        result = run_and_measure(policy, strength_factor=s)
        if result is None:
            print(f"{s}: too short to measure")
            continue
        r = result
        print(f"{r['strength_factor']}, {r['steps_survived']}, {r['distance']:.2f}, "
              f"{r['cadence_per_1000steps']:.2f}, {r['stride_length_approx']:.3f}, "
              f"{r['cv_stride_duration_R']:.3f}, {r['cv_stride_duration_L']:.3f}, "
              f"{r['double_support_pct']:.1f}, {r['single_support_pct']:.1f}")
        log_result(results_path, [r[k] for k in header], header)