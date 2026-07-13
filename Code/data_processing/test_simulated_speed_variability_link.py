"""
Tests whether our ALREADY-COLLECTED simulated combo data shows the same
speed-vs-stride-variability relationship found in real GaitNDD patients
(strongest in Parkinson's: r=-0.731). Uses existing CSVs -- no new
simulation needed.
"""
import pandas as pd
import numpy as np

print("=== Combo A & B (strength/vmax-based) ===")
ab = pd.read_csv("Results/all_combos_unified_v2.csv")
ab = ab[ab["confidence"] == "ok"].copy()
ab["speed"] = ab["distance"] / ab["steps_survived"]

for combo in ab["combo_name"].unique():
    subset = ab[ab["combo_name"] == combo].dropna(subset=["speed", "cv_stride_dur"])
    if len(subset) > 2:
        corr = subset["speed"].corr(subset["cv_stride_dur"])
        print(f"{combo}: speed vs cv_stride_dur correlation = {corr:.3f} (n={len(subset)})")

print("\n=== Combo C (feedback-delay/noise-based) ===")
c = pd.read_csv("Results/timing_noise_combo_full.csv")
c = c[c["confidence"] == "ok"].copy()
c["speed"] = c["distance"] / c["steps_survived"]

corr_c = c["speed"].corr(c["cv_stride_dur"])
print(f"combo_c_feedback_delay_noise: speed vs cv_stride_dur correlation = {corr_c:.3f} (n={len(c)})")

print("\n=== Raw data for visual inspection ===")
print("\nCombo A/B:")
print(ab[["combo_name", "speed", "cv_stride_dur"]].sort_values("speed").to_string(index=False))
print("\nCombo C:")
print(c[["param1_val", "param2_val", "speed", "cv_stride_dur"]].sort_values("speed").to_string(index=False))