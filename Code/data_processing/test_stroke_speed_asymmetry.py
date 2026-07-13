"""
Tests whether our already-collected Combo B (stroke-inspired) data shows
a speed-asymmetry relationship comparable to real stroke patients.
Real literature (systematic review/meta-analysis, 2025) reports
temporospatial asymmetry correlates with gait speed at |r| = 0.72-0.94
in real stroke survivors -- a strong, well-established clinical finding.
Uses only existing CSV data, no new simulation.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("Results/all_combos_unified_v2.csv")
stroke = df[(df["combo_name"] == "combo_b_stroke_unilateral") & (df["confidence"] == "ok")].copy()

print(f"Usable stroke combo rows: {len(stroke)}\n")

# Speed = distance / steps survived (same definition used throughout project)
stroke["speed"] = stroke["distance"] / stroke["steps_survived"]

# Stance asymmetry = absolute difference between the two sides -- a
# direct, simple asymmetry measure matching what real studies use
stroke["stance_asymmetry"] = (stroke["r_stance_pct"] - stroke["l_stance_pct"]).abs()

print("=== Raw data ===")
print(stroke[["param1_val", "param2_val", "speed", "stance_asymmetry", "double_support_pct"]].to_string(index=False))

print("\n=== Correlation: speed vs stance asymmetry ===")
corr_asym = stroke["speed"].corr(stroke["stance_asymmetry"])
print(f"r = {corr_asym:.3f} (n={len(stroke)})")
print("Real stroke literature range: |r| = 0.72-0.94 (systematic review/meta-analysis)")

print("\n=== Correlation: speed vs double-support % (secondary check) ===")
corr_ds = stroke["speed"].corr(stroke["double_support_pct"])
print(f"r = {corr_ds:.3f} (n={len(stroke)})")