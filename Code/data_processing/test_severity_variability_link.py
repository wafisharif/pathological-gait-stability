"""
Tests whether the "unstable subgroup" pattern found in stride-time CV
is explained by disease severity/duration or gait speed, using the
real GaitNDD subject-description.txt metadata (hand-transcribed here to
avoid parsing errors from the file's inconsistent spacing/MISSING values).
"""
import pandas as pd
import numpy as np

# Hand-transcribed directly from subject-description.txt, verified against
# the raw file. None = MISSING in the original file.
metadata = [
    ("control1", "Healthy", 57, "f", 1.33, 0), ("control2", "Healthy", 22, "m", 1.47, 0),
    ("control3", "Healthy", 23, "f", 1.44, 0), ("control4", "Healthy", 52, "f", 1.54, 0),
    ("control5", "Healthy", 47, "f", 1.54, 0), ("control6", "Healthy", 30, "f", 1.26, 0),
    ("control7", "Healthy", 22, "f", 1.54, 0), ("control8", "Healthy", 22, "f", 1.33, 0),
    ("control9", "Healthy", 32, "f", 1.47, 0), ("control10", "Healthy", 38, "f", 1.4, 0),
    ("control11", "Healthy", 69, "f", 0.91, 0), ("control12", "Healthy", 74, "m", 1.26, 0),
    ("control13", "Healthy", 61, "f", 1.33, 0), ("control14", "Healthy", 20, "f", 1.33, 0),
    ("control15", "Healthy", 20, "f", 1.19, 0), ("control16", "Healthy", 40, "f", 1.33, 0),
    ("hunt1", "Huntingtons", 42, "m", 1.68, 8), ("hunt2", "Huntingtons", 41, "f", 1.05, 11),
    ("hunt3", "Huntingtons", 66, "f", 1.05, 4), ("hunt4", "Huntingtons", 47, "f", 1.4, 2),
    ("hunt5", "Huntingtons", 36, "m", 1.82, 10), ("hunt6", "Huntingtons", 41, "f", 1.54, 8),
    ("hunt7", "Huntingtons", 71, "m", 1.05, 2), ("hunt8", "Huntingtons", 53, "f", 1.26, 9),
    ("hunt9", "Huntingtons", 54, "f", 1.26, 12), ("hunt10", "Huntingtons", 47, "f", 1.05, 4),
    ("hunt11", "Huntingtons", 33, "m", 1.26, 11), ("hunt12", "Huntingtons", 47, "m", 1.19, 8),
    ("hunt13", "Huntingtons", 40, "f", 0.56, 5), ("hunt14", "Huntingtons", 36, "f", 1.4, 12),
    ("hunt15", "Huntingtons", 34, "f", 0.56, 3), ("hunt16", "Huntingtons", 70, "m", 0.56, 5),
    ("hunt17", "Huntingtons", 29, "f", 1.19, 12), ("hunt18", "Huntingtons", 54, "f", 0.98, 2),
    ("hunt19", "Huntingtons", 59, "f", 0.98, 1), ("hunt20", "Huntingtons", 33, "f", None, 9),
    ("park1", "Parkinsons", 77, "m", 0.98, 4), ("park2", "Parkinsons", 44, "f", 1.26, 1.5),
    ("park3", "Parkinsons", 80, "m", 0.98, 2), ("park4", "Parkinsons", 74, "f", 0.91, 3.5),
    ("park5", "Parkinsons", 75, "m", 1.05, 2), ("park6", "Parkinsons", 53, "m", 1.33, 2),
    ("park7", "Parkinsons", 64, "f", 0.91, 4), ("park8", "Parkinsons", 64, "m", 0.84, 4),
    ("park9", "Parkinsons", 68, "m", 1.05, 1.5), ("park10", "Parkinsons", 60, "m", 1.19, 3),
    ("park11", "Parkinsons", 74, "m", 0.5, 3), ("park12", "Parkinsons", 57, "f", 0.98, 3),
    ("park13", "Parkinsons", 79, "f", 0.84, 3), ("park14", "Parkinsons", 57, "m", 0.98, 3),
    ("park15", "Parkinsons", 76, "m", 1.19, 2.5),
    ("als1", "ALS", 68, "m", 1.302, 1), ("als2", "ALS", 63, "m", 1.219, 14),
    ("als3", "ALS", 70, "f", 0.853, 13), ("als4", "ALS", 70, "f", None, 54),
    ("als5", "ALS", 36, "m", None, 5.5), ("als6", "ALS", 43, "m", 0.77, 17),
    ("als7", "ALS", 65, "m", 1.302, 9), ("als8", "ALS", 51, "m", 1.085, 3),
    ("als9", "ALS", 50, "m", 0.899, 54), ("als10", "ALS", 40, "f", 1.219, 14.5),
    ("als11", "ALS", 39, "m", 1.283, 7), ("als12", "ALS", 62, "m", 0.831, 12),
    ("als13", "ALS", 66, "f", 0.832, 34),
]

meta_df = pd.DataFrame(metadata, columns=["subject_id", "group", "age", "gender",
                                            "gait_speed", "duration_severity"])

cv_df = pd.read_csv("Results/gaitndd_stride_variability.csv")

merged = pd.merge(cv_df, meta_df, on=["subject_id", "group"], how="inner")
print(f"Merged {len(merged)} subjects (expected 64)\n")

print("=== Correlation: CV vs. duration/severity, per disease group ===")
for group in ["ALS", "Parkinsons", "Huntingtons"]:
    subset = merged[merged["group"] == group].dropna(subset=["duration_severity", "cv_stride"])
    if len(subset) > 2:
        corr = subset["duration_severity"].corr(subset["cv_stride"])
        print(f"{group}: correlation = {corr:.3f} (n={len(subset)})")

print("\n=== Correlation: CV vs. gait speed, per disease group ===")
for group in ["ALS", "Parkinsons", "Huntingtons"]:
    subset = merged[merged["group"] == group].dropna(subset=["gait_speed", "cv_stride"])
    if len(subset) > 2:
        corr = subset["gait_speed"].corr(subset["cv_stride"])
        print(f"{group}: correlation = {corr:.3f} (n={len(subset)})")

print("\n=== ALS subjects, full detail, sorted by CV ===")
als_detail = merged[merged["group"] == "ALS"].sort_values("cv_stride")
print(als_detail[["subject_id", "cv_stride", "duration_severity", "gait_speed", "age"]].to_string(index=False))

merged.to_csv("Results/gaitndd_cv_with_metadata.csv", index=False)
print("\nSaved merged dataset to Results/gaitndd_cv_with_metadata.csv")