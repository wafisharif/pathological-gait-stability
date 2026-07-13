"""
Consolidates all perturbation-testing results (Healthy, ALS, Stroke,
Feedback-delay; both sagittal and lateral) into one clean summary table
for the paper. Reports a clean threshold where the data shows one, and
explicitly flags "non-monotonic / complex stability boundary" where it
does not, rather than forcing a single number onto real dynamical
complexity (per Miles's guidance: document actual findings honestly,
don't oversimplify to force a cleaner story than the data supports).
"""
import pandas as pd

summary = [
    {"combo": "Healthy (0.85 strength)", "direction": "Sagittal",
     "threshold_desc": "Recovers <=400N, fails >=500N", "clean_threshold_N": 450, "pct_BW": "~55-60%"},
    {"combo": "Healthy (0.85 strength)", "direction": "Lateral",
     "threshold_desc": "Recovers <=550N, fails >=600N", "clean_threshold_N": 575, "pct_BW": "~70-75%"},
    {"combo": "ALS-inspired (0.6 strength)", "direction": "Sagittal",
     "threshold_desc": "Non-monotonic (recovers at 400,500,600; fails at 550,650,700,800)",
     "clean_threshold_N": None, "pct_BW": None},
    {"combo": "ALS-inspired (0.6 strength)", "direction": "Lateral",
     "threshold_desc": "Fails at all tested levels >=400N", "clean_threshold_N": "<400", "pct_BW": "<54%"},
    {"combo": "Stroke-inspired (0.6 paretic)", "direction": "Sagittal",
     "threshold_desc": "Recovers <=250N, fails >=300N", "clean_threshold_N": 275, "pct_BW": "~34-40%"},
    {"combo": "Stroke-inspired (0.6 paretic)", "direction": "Lateral",
     "threshold_desc": "Non-monotonic, intermittent failures from 150N", "clean_threshold_N": None, "pct_BW": None},
    {"combo": "Feedback-delay-inspired (jitter=1.0)", "direction": "Sagittal",
     "threshold_desc": "Recovery rate declines smoothly 0.29->0.07 across 400-800N", "clean_threshold_N": None, "pct_BW": None},
    {"combo": "Feedback-delay-inspired (jitter=1.0)", "direction": "Lateral",
     "threshold_desc": "Recovery rate declines smoothly 0.13->0.00 across 400-800N", "clean_threshold_N": None, "pct_BW": None},
]

df = pd.DataFrame(summary)
df.to_csv("Results/perturbation_summary_table.csv", index=False)
print(df.to_string(index=False))
print("\nSaved to Results/perturbation_summary_table.csv")