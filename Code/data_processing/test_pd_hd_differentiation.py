"""
Tests whether Combo C's parameter space can be split into a
Parkinson's-like sub-path (jitter-dominant, low noise) and a
Huntington's-like sub-path (jitter + higher noise), each aligning
better with its respective real disease direction -- addressing the
"one mechanism representing two diseases" gap identified in review.
Uses only already-collected data (Results/timing_noise_combo_full.csv).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SIM_DT_SECONDS = 0.01
features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

# --- Real disease direction vectors (recomputed here for a self-contained script) ---
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

print("=== Real direction vectors (Healthy -> Disease) ===")
for d, v in real_vectors.items():
    print(f"{d}: {dict(zip(features, np.round(v, 3)))}")
print()

# --- Load Combo C grid, apply same standardization ---
combo_c = pd.read_csv("Results/timing_noise_combo_full.csv")
combo_c["mean_stride_dur"] = combo_c["mean_stride_dur"] * SIM_DT_SECONDS

combo_c_scaled = scaler.transform(combo_c[features].values)
for i, f in enumerate(features):
    combo_c[f + "_scaled"] = combo_c_scaled[:, i]

def get_point(jitter, noise):
    row = combo_c[(combo_c["param1_val"] == jitter) & (combo_c["param2_val"] == noise)]
    if len(row) == 0:
        return None
    return row[[f + "_scaled" for f in features]].values[0]

# --- Define two severity paths through the existing grid ---
pd_mild = get_point(0.5, 0.0)
pd_severe = get_point(1.5, 0.05)
hd_mild = get_point(0.5, 0.05)
hd_severe = get_point(1.5, 0.2)

pd_vector = pd_severe - pd_mild
hd_vector = hd_severe - hd_mild

print("=== Simulated severity-path vectors ===")
print(f"PD-like path (low noise, jitter 0.5->1.5): {dict(zip(features, np.round(pd_vector, 3)))}")
print(f"HD-like path (higher noise, jitter 0.5->1.5): {dict(zip(features, np.round(hd_vector, 3)))}")
print()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=== Cosine similarity: PD-like path vs real diseases ===")
for disease, real_vec in real_vectors.items():
    print(f"  vs {disease}: {cosine_sim(pd_vector, real_vec):.3f}")

print("\n=== Cosine similarity: HD-like path vs real diseases ===")
for disease, real_vec in real_vectors.items():
    print(f"  vs {disease}: {cosine_sim(hd_vector, real_vec):.3f}")

print("\n=== DIFFERENTIATION CHECK ===")
pd_path_pd_sim = cosine_sim(pd_vector, real_vectors["Parkinsons"])
pd_path_hd_sim = cosine_sim(pd_vector, real_vectors["Huntingtons"])
hd_path_pd_sim = cosine_sim(hd_vector, real_vectors["Parkinsons"])
hd_path_hd_sim = cosine_sim(hd_vector, real_vectors["Huntingtons"])

print(f"PD-like path prefers: {'Parkinsons' if pd_path_pd_sim > pd_path_hd_sim else 'Huntingtons'} "
      f"(PD={pd_path_pd_sim:.3f}, HD={pd_path_hd_sim:.3f})")
print(f"HD-like path prefers: {'Parkinsons' if hd_path_pd_sim > hd_path_hd_sim else 'Huntingtons'} "
      f"(PD={hd_path_pd_sim:.3f}, HD={hd_path_hd_sim:.3f})")
print("\nGenuine differentiation would show: PD-like path prefers Parkinsons, HD-like path prefers Huntingtons")