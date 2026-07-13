"""
PCA comparison: simulated impairment combos vs. real GaitNDD population
data, in the same gait-feature space. This is the core BigData/data-mining
component of the project.

Per project framing: this compares combo "profiles" against real disease
profiles to see which mechanism's output pattern most resembles which
real population -- NOT a claim that any combo "is" a disease controller.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Load simulated combo data ---
sim = pd.read_csv("Results/all_combos_unified_v2.csv")
sim_ok = sim[sim["confidence"] == "ok"].copy()  # only trustworthy rows

# --- CRITICAL FIX: convert simulated stride duration from simulation
# timesteps to seconds, so it's on the same scale as real GaitNDD data.
# MuJoCo timestep confirmed earlier in the project: dt = 0.01s per step.
SIM_DT_SECONDS = 0.01
sim_ok["mean_stride_dur"] = sim_ok["mean_stride_dur"] * SIM_DT_SECONDS

# --- Real GaitNDD population values (extracted earlier from real dataset) ---
real_data = pd.DataFrame([
    {"combo_name": "REAL_Healthy",     "mean_stride_dur": 1.097, "double_support_pct": 28.225,
     "r_stance_pct": 64.394, "l_stance_pct": 63.828},
    {"combo_name": "REAL_ALS",         "mean_stride_dur": 1.469, "double_support_pct": 40.767,
     "r_stance_pct": 67.980, "l_stance_pct": 67.688},
    {"combo_name": "REAL_Parkinsons",  "mean_stride_dur": 1.140, "double_support_pct": 34.149,
     "r_stance_pct": 67.309, "l_stance_pct": 66.716},
    {"combo_name": "REAL_Huntingtons", "mean_stride_dur": 2.217, "double_support_pct": 30.171,
     "r_stance_pct": 68.316, "l_stance_pct": 65.143},
])

# --- Build the shared feature set (now genuinely comparable units) ---
features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

sim_features = sim_ok[["combo_name"] + features].copy()
real_features = real_data[["combo_name"] + features].copy()

print(f"Simulated stride duration now converted to seconds. Sample check:")
print(sim_features[["combo_name", "mean_stride_dur"]].head(3))
print()

combined = pd.concat([sim_features, real_features], ignore_index=True)
combined_clean = combined.dropna(subset=features)

print(f"Combined dataset for PCA: {len(combined_clean)} rows")
print(f"({len(sim_features)} simulated + {len(real_features)} real, "
      f"{len(combined) - len(combined_clean)} dropped for missing values)\n")

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

print("Feature loadings (how much each feature contributes to each PC):")
loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
print(loadings)
print()

# --- Compute distance from each simulated combo's average position to
# each real population's position, in PCA space -- this gives us a direct,
# quantifiable answer to "which combo looks most like which real disease"
print("=== Average PC position per group ===")
group_means = combined_clean.groupby("combo_name")[["PC1", "PC2"]].mean()
print(group_means)
print()

print("=== Distance from each simulated combo to each real population ===")
real_groups = [g for g in group_means.index if g.startswith("REAL_")]
sim_groups = [g for g in group_means.index if not g.startswith("REAL_")]

for sim_group in sim_groups:
    print(f"\n{sim_group}:")
    sim_point = group_means.loc[sim_group]
    distances = []
    for real_group in real_groups:
        real_point = group_means.loc[real_group]
        dist = np.sqrt((sim_point["PC1"] - real_point["PC1"])**2 +
                        (sim_point["PC2"] - real_point["PC2"])**2)
        distances.append((real_group, dist))
    distances.sort(key=lambda x: x[1])
    for real_group, dist in distances:
        print(f"  distance to {real_group}: {dist:.3f}")
    print(f"  --> closest real population: {distances[0][0]}")

combined_clean.to_csv("Results/pca_results.csv", index=False)
print("\nSaved full results to Results/pca_results.csv")