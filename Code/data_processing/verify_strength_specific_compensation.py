"""
Verifies the strength-specific compensation finding by directly comparing
speed at the mildest vs most severe trustworthy configuration within
each combo, rather than relying only on an aggregate correlation from
a small number of points.
"""
import pandas as pd

df = pd.read_csv("Results/all_combos_unified_v2.csv")
df = df[df["confidence"] == "ok"].copy()
df["speed"] = df["distance"] / df["steps_survived"]

print("=== Combo A (ALS): mildest vs most severe strength, vmax fixed at 1.0 ===")
a = df[(df["combo_name"] == "combo_a_ALS_strength_vmax") & (df["param2_val"] == 1.0)]
print(a[["param1_val", "speed"]].sort_values("param1_val").to_string(index=False))
print("(param1_val = strength; LOWER = more severe. Compensation confirmed if speed is HIGHER at lower strength.)\n")

print("=== Combo B (Stroke): mildest vs most severe paretic strength, vmax fixed at 1.0 ===")
b = df[(df["combo_name"] == "combo_b_stroke_unilateral") & (df["param2_val"] == 1.0)]
print(b[["param1_val", "speed"]].sort_values("param1_val").to_string(index=False))
print("(param1_val = paretic strength; LOWER = more severe.)\n")

print("=== Combo C (Feedback-delay): mildest vs most severe jitter, noise fixed at 0.0 ===")
c = pd.read_csv("Results/timing_noise_combo_full.csv")
c = c[c["confidence"] == "ok"]
c_fixed = c[c["param2_val"] == 0.0]
c_fixed = c_fixed.copy()
c_fixed["speed"] = c_fixed["distance"] / c_fixed["steps_survived"]
print(c_fixed[["param1_val", "speed"]].sort_values("param1_val").to_string(index=False))
print("(param1_val = jitter; HIGHER = more severe.)")