from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

FIXED_STRENGTH = 0.8


def get_strikes_and_ground(heights, foot_index):
    h = heights[:, foot_index]
    height_range = np.max(h) - np.min(h)
    if height_range <= 1e-8:
        return [], np.zeros(len(h), dtype=bool)
    threshold = np.min(h) + 0.25 * height_range
    on_ground = h < threshold
    strikes = [i for i in range(1, len(on_ground)) if on_ground[i] and not on_ground[i - 1]]
    return strikes, on_ground


def run_and_measure(policy, strength_factor, hold_steps, max_steps=1000):
    """
    hold_steps: how many simulation steps to repeat each policy decision
    before asking for a new one. 1 = normal (new action every step).
    Represents slowed neural processing speed -- a control-RATE limit,
    not a signal-magnitude or signal-dynamics distortion.
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
    current_action = None

    for step in range(max_steps):
        if step % hold_steps == 0:
            current_action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(current_action)
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
        return {"hold_steps": hold_steps, "steps_survived": steps_survived, "distance": distance,
                "measured_speed": measured_speed, "confidence": "too_short",
                "cadence_per_1000steps": float('nan'), "stride_length_approx": float('nan'),
                "double_support_pct": float('nan'), "single_support_pct": float('nan'), "n_total_strikes": 0}

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

    return {"hold_steps": hold_steps, "steps_survived": steps_survived, "distance": distance,
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

    hold_levels = [1, 2, 3, 4, 5, 6, 8, 10]
    results_path = "Results/als_strength_plus_holdrate.csv"
    header = ["strength_factor", "hold_steps", "steps_survived", "distance", "measured_speed",
               "confidence", "cadence_per_1000steps", "stride_length_approx",
               "double_support_pct", "single_support_pct", "n_total_strikes"]

    print(f"Fixed strength={FIXED_STRENGTH}")
    print("hold_steps, steps, dist, speed, conf, cadence, stride_len, dbl_supp%, single_supp%, n_strikes")
    for h in hold_levels:
        r = run_and_measure(policy, strength_factor=FIXED_STRENGTH, hold_steps=h)
        print(f"{r['hold_steps']}, {r['steps_survived']}, {r['distance']:.2f}, {r['measured_speed']:.4f}, "
              f"{r['confidence']}, {r['cadence_per_1000steps']:.2f}, {r['stride_length_approx']:.3f}, "
              f"{r['double_support_pct']:.1f}, {r['single_support_pct']:.1f}, {r['n_total_strikes']}")
        log_result(results_path, [FIXED_STRENGTH] + [r[k] for k in header[1:]], header)