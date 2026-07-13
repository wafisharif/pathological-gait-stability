"""
Final unification of Combo A, B, and C (relabeled from feedback-delay+noise
mechanism) into one PCA-ready dataset, all with identical, complete columns.
"""
import pandas as pd
import numpy as np

# --- Combo A/B (already has stance%/double-support%, single-episode measurements) ---
ab = pd.read_csv("Results/combo_all_consolidated.csv")
ab_unified = pd.DataFrame({
    "combo_name": ab["combo_name"],
    "param1_name": ab["param1_name"], "param1_val": ab["param1_val"],
    "param2_name": ab["param2_name"], "param2_val": ab["param2_val"],
    "n_strikes": ab["n_total_strikes"],
    "mean_stride_dur": ab["mean_stride_dur"],
    "cv_stride_dur": ab["cv_stride_dur"],
    "double_support_pct": ab["double_support_pct"],
    "r_stance_pct": ab["r_stance_pct"],
    "l_stance_pct": ab["l_stance_pct"],
    "steps_survived": ab["steps_survived"],
    "distance": ab["distance"],
    "confidence": ab["confidence"],
})

# --- Combo C (feedback delay + motor noise, relabeled, pooled-episode format, now complete) ---
c = pd.read_csv("Results/timing_noise_combo_full.csv")
c_unified = pd.DataFrame({
    "combo_name": c["combo_name"],  # already relabeled to combo_c_feedback_delay_noise
    "param1_name": c["param1_name"], "param1_val": c["param1_val"],
    "param2_name": c["param2_name"], "param2_val": c["param2_val"],
    "n_strikes": c["n_strikes"],
    "mean_stride_dur": c["mean_stride_dur"],
    "cv_stride_dur": c["cv_stride_dur"],
    "double_support_pct": c["double_support_pct"],
    "r_stance_pct": c["r_stance_pct"],
    "l_stance_pct": c["l_stance_pct"],
    "steps_survived": c["steps_survived"],
    "distance": c["distance"],
    "confidence": c["confidence"],
})

unified = pd.concat([ab_unified, c_unified], ignore_index=True)
unified.to_csv("Results/all_combos_unified_v2.csv", index=False)

print(f"Unified dataset: {len(unified)} rows, all columns complete\n")
print(f"Combo breakdown:\n{unified['combo_name'].value_counts()}\n")
print(f"Confidence breakdown:\n{unified['confidence'].value_counts()}\n")
print(f"Missing values per column:\n{unified.isna().sum()}")