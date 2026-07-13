"""
Systematically tests whether ALL THREE of our impairment combos show
the same limitation: speed does not reliably decrease with increasing
impairment severity, because the frozen policy prioritizes maintaining
forward velocity. This reframes three separate observations (ALS
cadence, Huntington's slowdown ceiling, stroke speed-asymmetry sign
mismatch) as one unified, cross-mechanism, methodological finding.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("Results/all_combos_unified_v2.csv")
df = df[df["confidence"] == "ok"].copy()
df["speed"] = df["distance"] / df["steps_survived"]

print("=== Speed vs. severity (param1_val, the primary severity dial) for each combo ===\n")
for combo in df["combo_name"].unique():
    subset = df[df["combo_name"] == combo].sort_values("param1_val")
    print(f"--- {combo} ---")
    print(subset[["param1_val", "param2_val", "speed"]].to_string(index=False))
    corr = subset["param1_val"].corr(subset["speed"])
    print(f"Correlation (severity dial vs speed): r = {corr:.3f}")
    print("(Positive r = MORE severe -> FASTER speed = confirms the compensation limitation)")
    print()