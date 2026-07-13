from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

REAL_HEALTHY_DS_PCT = 28.225250
REAL_ALS_DS_PCT = 40.766620


def measure_real_signatures(policy, strength_factor, vmax_factor=1.0, max_steps=1000, contact_threshold=10.0):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    scaled[:, 6] = original_gainprm[:, 6] * vmax_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    r_contact_force, l_contact_force = [], []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        r_force = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
        l_force = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]
        r_contact_force.append(r_force)
        l_contact_force.append(l_force)
        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    r_contact_force = np.array(r_contact_force)
    l_contact_force = np.array(l_contact_force)
    transient = min(30, steps_survived // 4)
    r_post = r_contact_force[transient:]
    l_post = l_contact_force[transient:]

    if len(r_post) < 5:
        return {"steps_survived": steps_survived, "confidence": "too_short", "double_support_pct": float('nan'),
                "n_total_strikes": 0, "distance": 0, "cadence_per_1000steps": float('nan')}

    r_on = r_post > contact_threshold
    l_on = l_post > contact_threshold
    both_on = r_on & l_on
    n = len(r_on)
    double_support_pct = np.sum(both_on) / n * 100

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    r_strikes = np.sum((r_on[1:]) & (~r_on[:-1]))
    l_strikes = np.sum((l_on[1:]) & (~l_on[:-1]))
    n_strikes = r_strikes + l_strikes
    cadence = n_strikes / n * 1000

    return {"steps_survived": steps_survived, "confidence": "ok" if n_strikes >= 6 else "low_confidence",
            "distance": distance, "double_support_pct": double_support_pct,
            "cadence_per_1000steps": cadence, "n_total_strikes": n_strikes}


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

    print(f"Target: Healthy DS%={REAL_HEALTHY_DS_PCT:.1f}, ALS DS%={REAL_ALS_DS_PCT:.1f}\n")

    strength_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    vmax_levels = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]

    results_path = "Results/full_grid_corrected.csv"
    header = ["strength_factor", "vmax_factor", "steps_survived", "confidence", "distance",
               "double_support_pct", "cadence_per_1000steps", "n_total_strikes", "gap_to_ALS"]

    print("strength, vmax, steps, conf, dist, ds_pct, cadence, n_strikes, gap_to_ALS")
    for s in strength_levels:
        for v in vmax_levels:
            r = measure_real_signatures(policy, strength_factor=s, vmax_factor=v)
            gap = REAL_ALS_DS_PCT - r['double_support_pct'] if not np.isnan(r['double_support_pct']) else float('nan')
            print(f"{s}, {v}, {r['steps_survived']}, {r['confidence']}, {r.get('distance',0):.2f}, "
                  f"{r['double_support_pct']:.2f}, {r['cadence_per_1000steps']:.2f}, {r['n_total_strikes']}, {gap:.2f}")
            log_result(results_path, [s, v, r['steps_survived'], r['confidence'], r.get('distance',0),
                        r['double_support_pct'], r['cadence_per_1000steps'], r['n_total_strikes'], gap], header)