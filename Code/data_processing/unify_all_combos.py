"""
Merges Combo A, B (from combo_all_consolidated.csv) and Combo C, D
(from timing_noise_combo_focused.csv) into one single, PCA-ready dataset
with identical columns across every row.
"""
import pandas as pd
import numpy as np

# --- Load Combo A/B (already has combo_name, param1/2, stance% columns) ---
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

# --- Load Combo C/D (timing jitter + noise, pooled-episode format) ---
cd = pd.read_csv("Results/timing_noise_combo_focused.csv")
cd_unified = pd.DataFrame({
    "combo_name": "combo_cd_timing_noise",
    "param1_name": "jitter_std", "param1_val": cd["jitter_std"],
    "param2_name": "noise_std", "param2_val": cd["noise_std"],
    "n_strikes": cd["n_total_strikes"],
    "mean_stride_dur": cd["mean_stride_dur"],
    "cv_stride_dur": cd["cv_stride_dur"],
    "double_support_pct": np.nan,  # not measured in the C/D script -- real gap, flagged not faked
    "r_stance_pct": np.nan,        # same -- flagged, not backfilled with guesses
    "l_stance_pct": np.nan,
    "steps_survived": cd["mean_steps_survived"],
    "distance": cd["mean_distance"],
    "confidence": cd["confidence"],
})

unified = pd.concat([ab_unified, cd_unified], ignore_index=True)
unified.to_csv("Results/all_combos_unified.csv", index=False)

print(f"Unified dataset: {len(unified)} rows")
print(f"Combo breakdown:\n{unified['combo_name'].value_counts()}")
print(f"\nConfidence breakdown:\n{unified['confidence'].value_counts()}")
print(f"\nRows with double_support_pct available: {unified['double_support_pct'].notna().sum()} / {len(unified)}")