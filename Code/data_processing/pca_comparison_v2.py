"""
Improved PCA comparison: simulated impairment combos vs. ALL individual
real GaitNDD subjects (not just 4 group averages), in the same gait-
feature space. This gives a much richer, statistically honest comparison
of natural real-patient variation vs. our simulated cluster.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SIM_DT_SECONDS = 0.01  # MuJoCo timestep, confirmed earlier in the project

# --- Load simulated combo data ---
sim = pd.read_csv("Results/all_combos_unified_v2.csv")
sim_ok = sim[sim["confidence"] == "ok"].copy()
sim_ok["mean_stride_dur"] = sim_ok["mean_stride_dur"] * SIM_DT_SECONDS

# --- Load ALL individual real GaitNDD subjects (not just group means) ---
real = pd.read_csv("Results/gaitndd_subject_summary.csv")

real_features = pd.DataFrame({
    "combo_name": "REAL_" + real["group"],  # e.g. REAL_ALS, REAL_Healthy, per subject
    "subject_id": real["subject_id"],
    "mean_stride_dur": real["mean_stride"],
    "double_support_pct": real["double_support_pct"],
    "r_stance_pct": real["right_stance_pct"],
    "l_stance_pct": real["left_stance_pct"],
})

features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

sim_features = sim_ok[["combo_name"] + features].copy()
sim_features["subject_id"] = "sim"
sim_features["is_real"] = False

real_features["is_real"] = True

combined = pd.concat([sim_features, real_features], ignore_index=True)
combined_clean = combined.dropna(subset=features)

print(f"Combined dataset for PCA: {len(combined_clean)} rows")
print(f"({len(sim_features)} simulated + {(combined_clean['is_real']).sum()} real individual subjects)\n")

print("Real subjects per group:")
print(real_features["combo_name"].value_counts())
print()

# --- Standardize and run PCA ---
X = combined_clean[features].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

combined_clean["PC1"] = components[:, 0]
combined_clean["PC2"] = components[:, 1]

print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"Total explained: {sum(pca.explained_variance_ratio_):.2%}\n")

loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
print("Feature loadings:")
print(loadings)
print()

# --- Compute the real-population spread (not just a single average point) ---
print("=== Real population spread in PCA space (mean +/- std) ===")
real_only = combined_clean[combined_clean["is_real"]]
for group in sorted(real_only["combo_name"].unique()):
    subset = real_only[real_only["combo_name"] == group]
    print(f"{group}: PC1 = {subset['PC1'].mean():.2f} +/- {subset['PC1'].std():.2f}, "
          f"PC2 = {subset['PC2'].mean():.2f} +/- {subset['PC2'].std():.2f}  (n={len(subset)})")
print()

# --- For each simulated combo, check whether it falls WITHIN any real
# population's natural range (not just closest average) ---
print("=== Does each simulated combo fall within real patient variation? ===")
sim_only = combined_clean[~combined_clean["is_real"]]

for sim_combo in sorted(sim_only["combo_name"].unique()):
    sim_subset = sim_only[sim_only["combo_name"] == sim_combo]
    sim_pc1_mean = sim_subset["PC1"].mean()
    sim_pc2_mean = sim_subset["PC2"].mean()
    print(f"\n{sim_combo} (avg PC1={sim_pc1_mean:.2f}, PC2={sim_pc2_mean:.2f}):")

    for group in sorted(real_only["combo_name"].unique()):
        subset = real_only[real_only["combo_name"] == group]
        pc1_min, pc1_max = subset["PC1"].min(), subset["PC1"].max()
        pc2_min, pc2_max = subset["PC2"].min(), subset["PC2"].max()
        within_pc1 = pc1_min <= sim_pc1_mean <= pc1_max
        within_pc2 = pc2_min <= sim_pc2_mean <= pc2_max
        status = "WITHIN RANGE" if (within_pc1 and within_pc2) else "outside range"
        print(f"  vs {group} (PC1: {pc1_min:.2f} to {pc1_max:.2f}, "
              f"PC2: {pc2_min:.2f} to {pc2_max:.2f}) --> {status}")

combined_clean.to_csv("Results/pca_results_v2_individual_subjects.csv", index=False)
print("\nSaved full results to Results/pca_results_v2_individual_subjects.csv")