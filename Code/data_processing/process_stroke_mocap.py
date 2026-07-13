"""
Extracts joint-angle and EMG asymmetry features from the real stroke
motion-capture dataset (Pside = paretic, Nside = non-paretic), per
subject, for comparison against our simulated stroke combo.

Honest scope note: this dataset's processed export does not include
ground-reaction-force or joint-moment data for stroke subjects (confirmed
0/50 subjects have GRF columns) -- only the able-bodied group has kinetics.
"Reduced paretic propulsion" therefore cannot be directly validated here;
ankle-angle range is used as a partial, approximate proxy, not a
substitute claim.
"""
import pandas as pd
import numpy as np

FILEPATH = "Datasets/stroke_mocap/MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx"

JOINT_ANGLES = ["AnkleAngles", "KneeAngles", "HipAngles", "PelvisAngles"]
EMG_MUSCLES = ["GASnorm", "RFnorm", "VLnorm", "BFnorm", "STnorm", "TAnorm", "ERSnorm"]


def process_subject(df, subject_id):
    row = {"subject_id": subject_id}

    for joint in JOINT_ANGLES:
        p_col = f"Pside_{joint}"
        n_col = f"Nside_{joint}"
        if p_col not in df.columns or n_col not in df.columns:
            continue

        p_vals = df[p_col].dropna()
        n_vals = df[n_col].dropna()
        if len(p_vals) == 0 or len(n_vals) == 0:
            continue

        p_rom = p_vals.max() - p_vals.min()
        n_rom = n_vals.max() - n_vals.min()
        p_peak = p_vals.max()
        n_peak = n_vals.max()

        row[f"{joint}_paretic_ROM"] = p_rom
        row[f"{joint}_nonparetic_ROM"] = n_rom
        rom_denom = p_rom + n_rom
        row[f"{joint}_ROM_asymmetry"] = (n_rom - p_rom) / rom_denom if rom_denom > 0 else np.nan

        peak_denom = abs(p_peak) + abs(n_peak)
        row[f"{joint}_peak_asymmetry"] = (n_peak - p_peak) / peak_denom if peak_denom > 0 else np.nan

    for muscle in EMG_MUSCLES:
        p_col = f"Pside_{muscle}"
        n_col = f"Nside_{muscle}"
        if p_col not in df.columns or n_col not in df.columns:
            continue

        p_vals = df[p_col].dropna()
        n_vals = df[n_col].dropna()
        if len(p_vals) == 0 or len(n_vals) == 0:
            continue

        p_mean = p_vals.mean()
        n_mean = n_vals.mean()
        denom = p_mean + n_mean
        row[f"{muscle}_asymmetry"] = (n_mean - p_mean) / denom if denom > 0 else np.nan

    return row


if __name__ == "__main__":
    xls = pd.ExcelFile(FILEPATH)
    subject_sheets = [s for s in xls.sheet_names if s.startswith("Sub")]
    print(f"Processing {len(subject_sheets)} stroke subjects...\n")

    results = []
    for sheet in subject_sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        row = process_subject(df, sheet)
        results.append(row)
        print(f"{sheet}: processed")

    results_df = pd.DataFrame(results)
    output_path = "Results/stroke_mocap_asymmetry_features.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved per-subject asymmetry features to {output_path}")
    print(f"\n=== Group means (across all {len(results_df)} stroke subjects) ===")
    numeric_cols = [c for c in results_df.columns if c != "subject_id"]
    print(results_df[numeric_cols].mean())