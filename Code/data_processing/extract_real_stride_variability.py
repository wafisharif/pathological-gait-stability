"""
Extracts per-subject stride-time coefficient of variation (CV) from the
raw GaitNDD .ts files -- something we have NOT computed before (earlier
processing only used subject-level MEANS, not variability). This
directly tests the literature's core claim that stride-time variability
is THE strongest biomarker separating Huntington's from Parkinson's
(Hausdorff et al. 2000), giving us a sharper, literature-grounded axis
than anything used in our PCA work so far.
"""
import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "Datasets/gaitndd"
COLUMNS = [
    "elapsed_time", "left_stride", "right_stride",
    "left_swing_s", "right_swing_s", "left_swing_pct", "right_swing_pct",
    "left_stance_s", "right_stance_s", "left_stance_pct", "right_stance_pct",
    "double_support_s", "double_support_pct"
]

def load_subject(filepath):
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)

if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, "*.ts"))
    results = []

    for filepath in all_files:
        filename = os.path.basename(filepath)
        subject_id = filename.replace(".ts", "")

        if subject_id.startswith("als"):
            group = "ALS"
        elif subject_id.startswith("hunt"):
            group = "Huntingtons"
        elif subject_id.startswith("park"):
            group = "Parkinsons"
        elif subject_id.startswith("control"):
            group = "Healthy"
        else:
            continue

        df = load_subject(filepath)

        # Combine left and right stride intervals into one series per
        # subject (each stride, either foot, is a real timing sample)
        all_strides = pd.concat([df["left_stride"], df["right_stride"]]).dropna()
        if len(all_strides) < 3:
            continue

        mean_stride = all_strides.mean()
        std_stride = all_strides.std()
        cv_stride = std_stride / mean_stride if mean_stride > 0 else np.nan

        results.append({
            "subject_id": subject_id, "group": group,
            "mean_stride": mean_stride, "std_stride": std_stride,
            "cv_stride": cv_stride, "n_strides": len(all_strides),
        })

    results_df = pd.DataFrame(results)
    output_path = "Results/gaitndd_stride_variability.csv"
    results_df.to_csv(output_path, index=False)

    print(f"Processed {len(results_df)} subjects\n")
    print("=== Real stride-time CV by group (mean +/- std across subjects) ===")
    group_stats = results_df.groupby("group")["cv_stride"].agg(["mean", "std", "min", "max", "count"])
    print(group_stats)
    print(f"\nSaved per-subject data to {output_path}")