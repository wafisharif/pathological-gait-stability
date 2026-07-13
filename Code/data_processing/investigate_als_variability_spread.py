"""
Investigates whether ALS's unusually large stride-CV spread is a real
biological finding or a statistical artifact of subjects with few
recorded strides.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("Results/gaitndd_stride_variability.csv")

print("=== Per-subject data, ALS group (sorted by CV) ===")
als = df[df["group"] == "ALS"].sort_values("cv_stride")
print(als[["subject_id", "cv_stride", "n_strides", "mean_stride"]].to_string(index=False))

print("\n=== Per-subject data, all other groups for comparison ===")
for group in ["Healthy", "Parkinsons", "Huntingtons"]:
    subset = df[df["group"] == group].sort_values("cv_stride")
    print(f"\n{group}:")
    print(subset[["subject_id", "cv_stride", "n_strides"]].to_string(index=False))

print("\n=== ARTIFACT CHECK: correlation between n_strides and cv_stride ===")
print("(if strongly negative, low-stride subjects are driving extreme CVs -- an artifact)")
for group in df["group"].unique():
    subset = df[df["group"] == group]
    if len(subset) > 2:
        corr = subset["n_strides"].corr(subset["cv_stride"])
        print(f"{group}: correlation = {corr:.3f} (n={len(subset)})")

print("\n=== MEDIAN comparison (robust to outliers, unlike mean) ===")
print(df.groupby("group")["cv_stride"].median())

print("\n=== Re-check group stats EXCLUDING subjects with fewer than 20 strides ===")
filtered = df[df["n_strides"] >= 20]
print(f"Subjects remaining after filter: {len(filtered)} / {len(df)}")
print(filtered.groupby("group")["cv_stride"].agg(["mean", "std", "min", "max", "count"]))