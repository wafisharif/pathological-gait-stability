"""
Rather than assuming which grid point is "HD-like," this searches ALL
12 already-collected Combo C configurations for the one that best
matches real Huntington's direction vector (dominated by a large
stride-time increase), computed relative to the mildest available
config as the reference point.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SIM_DT_SECONDS = 0.01
features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

real = pd.read_csv("Results/gaitndd_subject_summary.csv")
real_features = pd.DataFrame({
    "group": real["group"],
    "mean_stride_dur": real["mean_stride"],
    "double_support_pct": real["double_support_pct"],
    "r_stance_pct": real["right_stance_pct"],
    "l_stance_pct": real["left_stance_pct"],
})

scaler = StandardScaler()
real_scaled = scaler.fit_transform(real_features[features].values)
real_features_scaled = pd.DataFrame(real_scaled, columns=features)
real_features_scaled["group"] = real_features["group"].values
real_group_means = real_features_scaled.groupby("group")[features].mean()
healthy_point = real_group_means.loc["Healthy"].values

real_vectors = {
    disease: real_group_means.loc[disease].values - healthy_point
    for disease in ["Parkinsons", "Huntingtons"]
}

combo_c = pd.read_csv("Results/timing_noise_combo_full.csv")
combo_c["mean_stride_dur"] = combo_c["mean_stride_dur"] * SIM_DT_SECONDS
combo_c_scaled = scaler.transform(combo_c[features].values)
for i, f in enumerate(features):
    combo_c[f + "_scaled"] = combo_c_scaled[:, i]

print("=== Raw stride duration (seconds) across the full Combo C grid ===")
print(combo_c[["param1_val", "param2_val", "mean_stride_dur"]].sort_values("mean_stride_dur", ascending=False).to_string(index=False))
print()

# Use the mildest config as the reference "baseline" point
mild_row = combo_c[(combo_c["param1_val"] == 0.5) & (combo_c["param2_val"] == 0.0)]
mild_point = mild_row[[f + "_scaled" for f in features]].values[0]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=== Every grid config's direction (from mildest) vs real Huntington's vector ===")
results = []
for _, row in combo_c.iterrows():
    point = row[[f + "_scaled" for f in features]].values.astype(float)
    vec = point - mild_point
    vec_magnitude = np.linalg.norm(vec)
    if vec_magnitude < 0.3:  # skip near-zero, unreliable vectors
        sim_hd = float('nan')
        sim_pd = float('nan')
    else:
        sim_hd = cosine_sim(vec, real_vectors["Huntingtons"])
        sim_pd = cosine_sim(vec, real_vectors["Parkinsons"])
    results.append({"jitter": row["param1_val"], "noise": row["param2_val"],
                      "vector_magnitude": vec_magnitude,
                      "sim_to_HD": sim_hd, "sim_to_PD": sim_pd})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_hd = results_df.loc[results_df["sim_to_HD"].idxmax()]
print(f"\nBest-matching config for Huntington's direction: jitter={best_hd['jitter']}, "
      f"noise={best_hd['noise']}, similarity={best_hd['sim_to_HD']:.3f}")