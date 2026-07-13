"""
Compares DIRECTION of change (mild->severe impairment) in our simulated
combos against DIRECTION of change (Healthy->Disease) in real GaitNDD
data, using cosine similarity in standardized feature space. This tests
whether our impairments push gait the same WAY as real disease, even
though absolute position (confirmed via 3 separate PCA analyses) does not
match real human gait space.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SIM_DT_SECONDS = 0.01
features = ["mean_stride_dur", "double_support_pct", "r_stance_pct", "l_stance_pct"]

# --- Real data: compute Healthy -> Disease vectors ---
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

real_vectors = {}
for disease in ["ALS", "Parkinsons", "Huntingtons"]:
    real_vectors[disease] = real_group_means.loc[disease].values - healthy_point

print("=== Real disease direction vectors (Healthy -> Disease), standardized units ===")
for disease, vec in real_vectors.items():
    print(f"{disease}: {dict(zip(features, np.round(vec, 3)))}")
print()

# --- Simulated data: compute mild -> severe vectors per combo ---
sim = pd.read_csv("Results/all_combos_unified_v2.csv")
sim_ok = sim[sim["confidence"] == "ok"].copy()
sim_ok["mean_stride_dur"] = sim_ok["mean_stride_dur"] * SIM_DT_SECONDS

# Use the SAME scaler fit on real data, so simulated points are on the
# same standardized scale as the real vectors above
sim_scaled = scaler.transform(sim_ok[features].values)
sim_ok_scaled = sim_ok.copy()
for i, f in enumerate(features):
    sim_ok_scaled[f] = sim_scaled[:, i]

# Define mild/severe reference rows per combo (least vs. most impaired
# trustworthy configuration within each combo's own parameter sweep)
combo_defs = {
    "combo_a_ALS_strength_vmax": {
        "mild": {"param1_val": 0.85, "param2_val": 1.0},
        "severe": {"param1_val": 0.6, "param2_val": 1.0},
    },
    "combo_b_stroke_unilateral": {
        "mild": {"param1_val": 0.8, "param2_val": 1.0},
        "severe": {"param1_val": 0.6, "param2_val": 1.0},
    },
    "combo_c_feedback_delay_noise": {
        "mild": {"param1_val": 0.5, "param2_val": 0.0},
        "severe": {"param1_val": 1.5, "param2_val": 0.2},
    },
}

sim_vectors = {}
print("=== Simulated combo direction vectors (mild -> severe), standardized units ===")
for combo_name, defs in combo_defs.items():
    subset = sim_ok_scaled[sim_ok_scaled["combo_name"] == combo_name]

    mild_row = subset[(subset["param1_val"] == defs["mild"]["param1_val"]) &
                        (subset["param2_val"] == defs["mild"]["param2_val"])]
    severe_row = subset[(subset["param1_val"] == defs["severe"]["param1_val"]) &
                          (subset["param2_val"] == defs["severe"]["param2_val"])]

    if len(mild_row) == 0 or len(severe_row) == 0:
        print(f"{combo_name}: could not find matching mild/severe rows, skipping")
        continue

    mild_point = mild_row[features].values[0]
    severe_point = severe_row[features].values[0]
    vec = severe_point - mild_point
    sim_vectors[combo_name] = vec
    print(f"{combo_name}: {dict(zip(features, np.round(vec, 3)))}")
print()

# --- Cosine similarity between each sim vector and each real vector ---
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=== Cosine similarity: simulated direction vs. real disease direction ===")
print("(1.0 = identical direction, 0.0 = unrelated/perpendicular, -1.0 = opposite direction)\n")
for combo_name, sim_vec in sim_vectors.items():
    print(f"{combo_name}:")
    sims = []
    for disease, real_vec in real_vectors.items():
        sim_score = cosine_sim(sim_vec, real_vec)
        sims.append((disease, sim_score))
    sims.sort(key=lambda x: x[1], reverse=True)
    for disease, score in sims:
        print(f"  vs {disease}: {score:.3f}")
    print(f"  --> most similar direction: {sims[0][0]} (cosine={sims[0][1]:.3f})\n")