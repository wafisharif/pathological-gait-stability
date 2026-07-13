"""
Consolidated Combo A (ALS-inspired: global strength + vmax) and
Combo B (stroke-inspired: unilateral strength + vmax), measured with
the same approach and output format as the Combo C/D timing+noise sweep,
so all four combos can be merged into one dataset for PCA later.
"""
from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

CONTACT_THRESHOLD = 10.0


def measure_episode(sim, env, policy, max_steps=1000):
    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]
    r_trace, l_trace = [], []
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        r_trace.append(sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0])
        l_trace.append(sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0])
        steps += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return np.array(r_trace), np.array(l_trace), steps, distance


def analyze_traces(r_trace, l_trace, steps):
    transient = min(30, steps // 4)
    r, l = r_trace[transient:], l_trace[transient:]
    n = len(r)
    if n < 10:
        return None

    r_on, l_on = r > CONTACT_THRESHOLD, l > CONTACT_THRESHOLD
    both = r_on & l_on

    r_strikes_idx = [i for i in range(1, len(r_on)) if r_on[i] and not r_on[i-1]]
    l_strikes_idx = [i for i in range(1, len(l_on)) if l_on[i] and not l_on[i-1]]
    n_strikes = len(r_strikes_idx) + len(l_strikes_idx)

    all_durations = []
    if len(r_strikes_idx) >= 2:
        all_durations.extend(np.diff(r_strikes_idx))
    if len(l_strikes_idx) >= 2:
        all_durations.extend(np.diff(l_strikes_idx))

    mean_dur = np.mean(all_durations) if len(all_durations) >= 2 else float('nan')
    cv_dur = (np.std(all_durations) / mean_dur) if len(all_durations) >= 2 and mean_dur > 0 else float('nan')

    double_support_pct = np.sum(both) / n * 100
    r_stance_pct = np.sum(r_on) / n * 100
    l_stance_pct = np.sum(l_on) / n * 100

    confidence = "ok" if n_strikes >= 10 else ("low_confidence" if n_strikes >= 2 else "too_short")

    return {
        "n_total_strikes": n_strikes, "mean_stride_dur": mean_dur, "cv_stride_dur": cv_dur,
        "double_support_pct": double_support_pct, "r_stance_pct": r_stance_pct,
        "l_stance_pct": l_stance_pct, "confidence": confidence,
    }


def log_result(filepath, row, header):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


HEADER = ["combo_name", "param1_name", "param1_val", "param2_name", "param2_val",
           "n_total_strikes", "mean_stride_dur", "cv_stride_dur",
           "double_support_pct", "r_stance_pct", "l_stance_pct",
           "steps_survived", "distance", "confidence"]
RESULTS_PATH = "Results/combo_all_consolidated.csv"


def run_combo_a(policy, strength_factor, vmax_factor):
    """ALS-inspired: global strength + global vmax, both muscle-capacity properties."""
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    scaled[:, 2] = original[:, 2] * strength_factor
    scaled[:, 6] = original[:, 6] * vmax_factor
    sim.model.actuator_gainprm[:] = scaled

    r_trace, l_trace, steps, distance = measure_episode(sim, env, policy)
    env.close()

    result = analyze_traces(r_trace, l_trace, steps)
    if result is None:
        row = ["combo_a_ALS_strength_vmax", "strength", strength_factor, "vmax", vmax_factor,
               0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
               steps, distance, "too_short"]
    else:
        row = ["combo_a_ALS_strength_vmax", "strength", strength_factor, "vmax", vmax_factor,
               result["n_total_strikes"], result["mean_stride_dur"], result["cv_stride_dur"],
               result["double_support_pct"], result["r_stance_pct"], result["l_stance_pct"],
               steps, distance, result["confidence"]]
    print(", ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in row))
    log_result(RESULTS_PATH, row, HEADER)


def run_combo_b(policy, paretic_strength, paretic_vmax, paretic_side="left"):
    """Stroke-inspired: unilateral strength + unilateral vmax, same side."""
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    idx = slice(40, 80) if paretic_side == "left" else slice(0, 40)
    scaled[idx, 2] = original[idx, 2] * paretic_strength
    scaled[idx, 6] = original[idx, 6] * paretic_vmax
    sim.model.actuator_gainprm[:] = scaled

    r_trace, l_trace, steps, distance = measure_episode(sim, env, policy)
    env.close()

    result = analyze_traces(r_trace, l_trace, steps)
    if result is None:
        row = ["combo_b_stroke_unilateral", "paretic_strength", paretic_strength,
               "paretic_vmax", paretic_vmax, 0, float('nan'), float('nan'),
               float('nan'), float('nan'), float('nan'), steps, distance, "too_short"]
    else:
        row = ["combo_b_stroke_unilateral", "paretic_strength", paretic_strength,
               "paretic_vmax", paretic_vmax, result["n_total_strikes"], result["mean_stride_dur"],
               result["cv_stride_dur"], result["double_support_pct"], result["r_stance_pct"],
               result["l_stance_pct"], steps, distance, result["confidence"]]
    print(", ".join(str(x) if not isinstance(x, float) else f"{x:.3f}" for x in row))
    log_result(RESULTS_PATH, row, HEADER)


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("=== COMBO A: ALS-inspired (global strength + vmax) ===")
    print("combo, param1, val1, param2, val2, n_strikes, mean_dur, cv_dur, ds%, r_stance%, l_stance%, steps, dist, conf")
    for strength in [0.85, 0.8, 0.75, 0.6]:
        for vmax in [1.0, 0.8, 0.6]:
            run_combo_a(policy, strength, vmax)

    print("\n=== COMBO B: Stroke-inspired (unilateral strength + vmax, left paretic) ===")
    for p_strength in [0.8, 0.6, 0.5, 0.4]:
        for p_vmax in [1.0, 0.8, 0.6]:
            run_combo_b(policy, p_strength, p_vmax, paretic_side="left")