from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os
import argparse

# ============================================================
# GaitNDD REFERENCE VALUES
# ============================================================
REAL = {
    "Healthy": {
        "double_support_pct": 28.225,
        "left_stance_pct":    63.828,
        "right_stance_pct":   64.394,
        "mean_stride_time_s": 1.097,
    },
    "ALS": {
        "double_support_pct": 40.767,
        "left_stance_pct":    67.688,
        "right_stance_pct":   67.980,
        "mean_stride_time_s": 1.469,
    },
    "Parkinsons": {
        "double_support_pct": 34.149,
        "left_stance_pct":    66.716,
        "right_stance_pct":   67.309,
        "mean_stride_time_s": 1.140,
    },
    "Huntingtons": {
        "double_support_pct": 30.171,
        "left_stance_pct":    65.143,
        "right_stance_pct":   68.316,
        "mean_stride_time_s": 2.217,
    },
}

PLANTARFLEXOR_NAMES = ['soleus', 'gaslat', 'gasmed']
HIP_EXTENSOR_NAMES  = ['glmax', 'semimem', 'semiten', 'bflh']

# ============================================================
# CONTROLLER CONFIGURATIONS
# ============================================================
def setup_healthy(sim):
    original = sim.model.actuator_gainprm.copy()
    scaled   = original.copy()
    scaled[:, 2] = original[:, 2] * 0.85
    sim.model.actuator_gainprm[:] = scaled


def setup_als_v1(sim):
    muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
    pf_idx  = [i for i, n in enumerate(muscle_names)
                if any(n.startswith(p) for p in PLANTARFLEXOR_NAMES)]
    hip_idx = [i for i, n in enumerate(muscle_names)
                if any(n.startswith(p) for p in HIP_EXTENSOR_NAMES)]
    original = sim.model.actuator_gainprm.copy()
    scaled   = original.copy()
    scaled[:, 2]         = original[:, 2] * 0.6
    scaled[pf_idx,  2]   = original[pf_idx,  2] * 0.6 * 0.6
    scaled[hip_idx, 2]   = original[hip_idx, 2] * 0.6 * 0.6
    sim.model.actuator_gainprm[:] = scaled

def setup_stroke_v1(sim, paretic_side="left", paretic_strength=0.4):
    """
    Stroke-like unilateral weakness: one side (paretic) significantly
    weaker than the other (non-paretic, left at full strength).
    Confirmed muscle index split: 0-39 = right leg, 40-79 = left leg.
    Reproduces the unilateral-weakness / asymmetry signature of stroke gait
    -- NOT a full stroke replica.
    """
    original = sim.model.actuator_gainprm.copy()
    scaled   = original.copy()
    if paretic_side == "left":
        scaled[40:80, 2] = original[40:80, 2] * paretic_strength
    elif paretic_side == "right":
        scaled[0:40, 2]  = original[0:40, 2]  * paretic_strength
    sim.model.actuator_gainprm[:] = scaled


CONTROLLERS = {
    "healthy": (setup_healthy,   "Simulated Healthy (0.85x)",  "Healthy"),
    "als":     (setup_als_v1,    "ALS Controller v1",          "ALS"),
    "stroke":  (lambda sim: setup_stroke_v1(sim, paretic_side="left", paretic_strength=0.4),
                "Stroke Controller v1 (L paretic)", "Healthy"),
}


# ============================================================
# MEASUREMENT
# ============================================================
def measure(policy, label, setup_fn, max_steps=1000, contact_threshold=10.0):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    if setup_fn:
        setup_fn(sim)

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]
    r_trace, l_trace  = [], []
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        r_trace.append(sim.data.sensor('r_foot').data[0] +
                       sim.data.sensor('r_toes').data[0])
        l_trace.append(sim.data.sensor('l_foot').data[0] +
                       sim.data.sensor('l_toes').data[0])
        steps += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    r = np.array(r_trace);  l = np.array(l_trace)
    transient = min(30, steps // 4)
    r, l = r[transient:], l[transient:]
    n = len(r)
    if n < 10:
        return None

    r_on, l_on = r > contact_threshold, l > contact_threshold
    both    = r_on & l_on

    def stride_time(on):
        strikes = [i for i in range(1, len(on)) if on[i] and not on[i-1]]
        if len(strikes) < 2:
            return float('nan')
        return float(np.mean(np.diff(strikes) * 0.01))

    distance = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    n_strikes = (np.sum((r_on[1:]) & (~r_on[:-1])) +
                 np.sum((l_on[1:]) & (~l_on[:-1])))

    return {
        "label":                label,
        "steps_survived":       steps,
        "n_strikes":            int(n_strikes),
        "distance_m":           round(distance, 3),
        "double_support_pct":   round(np.sum(both) / n * 100, 2),
        "right_stance_pct":     round(np.sum(r_on)  / n * 100, 2),
        "left_stance_pct":      round(np.sum(l_on)  / n * 100, 2),
        "mean_stride_time_s":   round(stride_time(r_on), 3),
    }


def log_to_csv(result, population_target, filepath="Results/benchmark_results.csv"):
    header = ["label", "target_population", "steps_survived", "n_strikes",
               "distance_m", "double_support_pct", "right_stance_pct",
               "left_stance_pct", "mean_stride_time_s"]
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([result[k] if k != "target_population"
                         else population_target for k in
                         ["label"] + ["target_population"] + header[2:]])


def print_report(result, real_target_name, sim_healthy_result=None):
    real_target  = REAL.get(real_target_name, {})
    real_healthy = REAL.get("Healthy", {})
    metrics = [
        ("double_support_pct", "Double-support %"),
        ("right_stance_pct",   "Right stance %"),
        ("left_stance_pct",    "Left stance %"),
        ("mean_stride_time_s", "Stride time (s)"),
    ]
    print(f"\n{'='*75}")
    print(f"BENCHMARK: {result['label']}  (target: {real_target_name})")
    print(f"{'='*75}")
    print(f"{'Metric':<25} {'RealHlthy':>10} {'RealALS':>10} {'SimHlthy':>10} {'SimALS':>10} {'Dir OK?':>8}")
    print(f"{'-'*75}")
    for key, label in metrics:
        rh  = real_healthy.get(key, float('nan'))
        ra  = real_target.get(key, float('nan'))
        sh  = sim_healthy_result.get(key, float('nan')) if sim_healthy_result else float('nan')
        sa  = result.get(key, float('nan'))
        real_dir = ra > rh
        sim_dir  = sa > sh
        match = "✅" if real_dir == sim_dir else "❌"
        print(f"  {label:<23} {rh:>10.2f} {ra:>10.2f} {sh:>10.2f} {sa:>10.2f} {match:>8}")
    print(f"{'-'*75}")
    print(f"  Steps: {result['steps_survived']}, Strikes: {result['n_strikes']}, Distance: {result['distance_m']}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="als",
                        choices=list(CONTROLLERS.keys()))
    args = parser.parse_args()

    setup_fn, label, target_pop = CONTROLLERS[args.controller]

    env    = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    # always run healthy reference first (uses 0.85x strength for stability)
    print("Running healthy reference...")
    sim_healthy = measure(policy, "Sim Healthy", setup_healthy)

    print(f"Benchmarking: {label}")
    result = measure(policy, label, setup_fn)

    if result:
        print_report(result, target_pop, sim_healthy_result=sim_healthy)
        log_to_csv(result, target_pop)
        print(f"\nLogged to Results/benchmark_results.csv")
