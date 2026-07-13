from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

FIXED_STRENGTH = 0.8  # known-good, full-survival, validated strength level


def get_strikes_and_ground(heights, foot_index):
    h = heights[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return [], np.zeros(len(h), dtype=bool)
    threshold = np.min(h) + 0.25 * height_range
    on_ground = h < threshold
    strikes = [i for i in range(1, len(on_ground)) if on_ground[i] and not on_ground[i - 1]]
    return strikes, on_ground


def run_and_measure(policy, strength_factor, action_cap, max_steps=1000):
    """
    action_cap: maximum absolute value allowed for any muscle command.
    1.0 = no cap (full range). Lower = ceiling on how strongly any muscle
    can be commanded to contract, independent of its force CAPACITY
    (strength), which stays controlled separately via gainprm.
    """
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
        action = policy(obs).copy()
        action = np.clip(action, -action_cap, action_cap)  # the new, separate mechanism

        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = env.unwrapped.get_obs_dict(sim)
        feet_heights.append(np.array(obs_dict["feet_heights"]).copy())
        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    feet_heights = np.array(feet_heights)
    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    measured_speed = distance / max(steps_survived, 1)

    transient = min(30, steps_survived // 4)
    feet_heights_post = feet_heights[transient:]

    if len(feet_heights_post) < 5:
        return {"strength_factor": strength_factor, "action_cap": action_cap,
                "steps_survived": steps_survived, "distance": distance,
                "measured_speed": measured_speed, "confidence": "too_short",
                "cadence_per_1000steps": float('nan'), "stride_length_approx": float('nan'),
                "double_support_pct": float('nan'), "single_support_pct": float('nan'),
                "n_total_strikes": 0}

    strikes_r, on_ground_r = get_strikes_and_ground(feet_heights_post, foot_index=0)
    strikes_l, on_ground_l = get_strikes_and_ground(feet_heights_post, foot_index=1)
    n_strikes = len(strikes_r) + len(strikes_l)
    n = len(on_ground_r)

    cadence_normalized = n_strikes / n * 1000
    stride_len = distance / max(n_strikes, 1)

    both_on = on_ground_r & on_ground_l
    only_r = on_ground_r & ~on_ground_l
    only_l = on_ground_l & ~on_ground_r
    double_support_pct = np.sum(both_on) / n * 100
    single_support_pct = (np.sum(only_r) + np.sum(only_l)) / n * 100

    confidence = "ok" if n_strikes >= 4 else "low_confidence"

    return {"strength_factor": strength_factor, "action_cap": action_cap,
            "steps_survived": steps_survived, "distance": distance,
            "measured_speed": measured_speed, "confidence": confidence,
            "cadence_per_1000steps": cadence_normalized, "stride_length_approx": stride_len,
            "double_support_pct": double_support_pct, "single_support_pct": single_support_pct,
            "n_total_strikes": n_strikes}


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

    action_caps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    results_path = "Results/als_strength_plus_cap.csv"
    header = ["strength_factor", "action_cap", "steps_survived", "distance", "measured_speed",
               "confidence", "cadence_per_1000steps", "stride_length_approx",
               "double_support_pct", "single_support_pct", "n_total_strikes"]

    print(f"Fixed strength={FIXED_STRENGTH}")
    print("action_cap, steps, dist, speed, conf, cadence, stride_len, dbl_supp%, single_supp%, n_strikes")
    for cap in action_caps:
        r = run_and_measure(policy, strength_factor=FIXED_STRENGTH, action_cap=cap)
        print(f"{r['action_cap']}, {r['steps_survived']}, {r['distance']:.2f}, {r['measured_speed']:.4f}, "
              f"{r['confidence']}, {r['cadence_per_1000steps']:.2f}, {r['stride_length_approx']:.3f}, "
              f"{r['double_support_pct']:.1f}, {r['single_support_pct']:.1f}, {r['n_total_strikes']}")
        log_result(results_path, [r[k] for k in header], header)