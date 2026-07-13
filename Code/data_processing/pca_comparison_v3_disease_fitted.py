"""
PCA fitted ONLY on real GaitNDD patient data (so the axes reflect real
disease differences, not the sim-vs-real gap), then simulated combos are
projected onto those same axes afterward. This answers "where would our
simulated impairments land on a map drawn purely from real disease
variation" -- a more targeted question than mixing everything together.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SIM_DT_SECONDS = 0.01

# --- Load real GaitNDD individual subjects ---
real = pd.read_csv("Results/gaitndd_subject_summary.csv")
real_features = pd.DataFrame({
    "combo_name": "REAL_" + real["group"],
    "mean_stride_dur": real["mean_stride"],
    "double_support_pct": real["double_support_pct"],
    "r_stance_pct": real["right_stance_pct"],
    "l_stance_pct": real["left_stance_pct"],
})

# --- Load simulated combo data ---
sim = pd.read_csv("Results/all_combos_unified_v2.csv")
sim_ok = sim[sim["confidence"] == "ok"].copy()
sim_ok["mean_stride_dur"] = sim_ok["mean_stride_dur"] * SIM_DT_SECONDS
sim_features = sim_ok[["combo_name", "mean_stride_dur", "double_support_pct",
                        "r_stance_pct", "l_stance_pct"]].copy()

features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

# --- KEY DIFFERENCE: fit the scaler AND PCA using ONLY real data ---
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(real_features[features].values)

pca = PCA(n_components=2)
real_components = pca.fit_transform(X_real_scaled)

real_features["PC1"] = real_components[:, 0]
real_features["PC2"] = real_components[:, 1]

print("=== PCA fitted on REAL data only ===")
print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
      f"PC2={pca.explained_variance_ratio_[1]:.2%}")
print(f"Total explained: {sum(pca.explained_variance_ratio_):.2%}\n")

loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
print("Feature loadings (based on real disease differences only):")
print(loadings)
print()

print("=== Real population spread on this disease-based map ===")
for group in sorted(real_features["combo_name"].unique()):
    subset = real_features[real_features["combo_name"] == group]
    print(f"{group}: PC1 = {subset['PC1'].mean():.2f} +/- {subset['PC1'].std():.2f}, "
          f"PC2 = {subset['PC2'].mean():.2f} +/- {subset['PC2'].std():.2f}  (n={len(subset)})")
print()

# --- Now project simulated data onto the SAME axes (transform only, no re-fit) ---
X_sim_scaled = scaler.transform(sim_features[features].values)  # uses real data's mean/std
sim_components = pca.transform(X_sim_scaled)  # uses real data's PCA directions

sim_features["PC1"] = sim_components[:, 0]
sim_features["PC2"] = sim_components[:, 1]

print("=== Where simulated combos land on the REAL disease map ===\n")
for sim_combo in sorted(sim_features["combo_name"].unique()):
    subset = sim_features[sim_features["combo_name"] == sim_combo]
    pc1_mean, pc2_mean = subset["PC1"].mean(), subset["PC2"].mean()
    print(f"{sim_combo}: avg PC1={pc1_mean:.2f}, PC2={pc2_mean:.2f}")

    distances = []
    for group in sorted(real_features["combo_name"].unique()):
        real_subset = real_features[real_features["combo_name"] == group]
        real_pc1, real_pc2 = real_subset["PC1"].mean(), real_subset["PC2"].mean()
        dist = np.sqrt((pc1_mean - real_pc1)**2 + (pc2_mean - real_pc2)**2)
        distances.append((group, dist))
    distances.sort(key=lambda x: x[1])
    for group, dist in distances:
        print(f"    distance to {group}: {dist:.3f}")
    print(f"    --> closest real population on this map: {distances[0][0]}\n")

# --- Save everything ---
real_features["is_real"] = True
sim_features["is_real"] = False
combined = pd.concat([real_features, sim_features], ignore_index=True)
combined.to_csv("Results/pca_results_v3_disease_fitted.csv", index=False)
print("Saved full results to Results/pca_results_v3_disease_fitted.csv")