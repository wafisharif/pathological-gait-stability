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
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)
    return df

def summarize_subject(df, subject_id, group):
    means = df[COLUMNS[1:]].mean()
    row = {"subject_id": subject_id, "group": group}
    row.update(means.to_dict())
    row["mean_stride"] = (df["left_stride"].mean() + df["right_stride"].mean()) / 2
    return row

if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, "*.ts"))
    print(f"Found {len(all_files)} .ts files")

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
            print(f"Unknown group for {subject_id}, skipping")
            continue

        df = load_subject(filepath)
        results.append(summarize_subject(df, subject_id, group))

    results_df = pd.DataFrame(results)

    output_path = "Results/gaitndd_subject_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved per-subject summary to {output_path}")

    print("\n=== GROUP MEANS ===")
    group_means = results_df.groupby("group")[["mean_stride", "double_support_pct",
                                                  "left_stance_pct", "right_stance_pct"]].mean()
    print(group_means)

    group_means.to_csv("Results/gaitndd_group_means.csv")
    print("\nSaved group means to Results/gaitndd_group_means.csv")