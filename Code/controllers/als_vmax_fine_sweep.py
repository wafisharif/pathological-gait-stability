from myosuite.utils import gym
import deprl
import numpy as np
import csv
import os

# Reuse the same functions from als_strength_plus_vmax.py
from als_strength_plus_vmax import run_and_measure, log_result

if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    strength = 0.8
    vmax_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]

    results_path = "Results/als_vmax_fine_sweep.csv"
    header = ["strength_factor", "vmax_factor", "steps_survived", "distance", "measured_speed",
               "confidence", "cadence_per_1000steps", "stride_length_approx",
               "double_support_pct", "single_support_pct", "n_total_strikes"]

    print(f"Fixed strength={strength}")
    print("vmax, steps, dist, speed, conf, cadence, stride_len, dbl_supp%, single_supp%, n_strikes")
    for vmax in vmax_levels:
        r = run_and_measure(policy, strength_factor=strength, vmax_factor=vmax)
        print(f"{vmax}, {r['steps_survived']}, {r['distance']:.2f}, {r['measured_speed']:.4f}, "
              f"{r['confidence']}, {r['cadence_per_1000steps']:.2f}, {r['stride_length_approx']:.3f}, "
              f"{r['double_support_pct']:.1f}, {r['single_support_pct']:.1f}, {r['n_total_strikes']}")
        log_result(results_path, [r[k] for k in header], header)