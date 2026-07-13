"""
Builds averaged, phase-indexed reference trajectories from the real
able-bodied motion-capture dataset, for use as reward targets in the
new physiologically-grounded cost function. Averages across many real
subjects rather than using a single person's stride, so the reference
represents typical healthy movement.
"""
import pandas as pd
import numpy as np

FILEPATH = "Datasets/stroke_mocap/MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx"
VARIABLES_TO_EXTRACT = ["HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles"]
N_SUBJECTS_TO_USE = 30  # a reasonable, fast-to-process subset; can expand later


def build_reference(filepath, variables, n_subjects=30):
    xls = pd.ExcelFile(filepath)
    subject_sheets = [s for s in xls.sheet_names if s.startswith("Sub")][:n_subjects]

    all_subjects_data = {var: [] for var in variables}

    for sheet in subject_sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        for var in variables:
            if var in df.columns:
                all_subjects_data[var].append(df[var].values)

    reference_curves = {}
    for var in variables:
        stacked = np.array(all_subjects_data[var])  # shape: (n_subjects, 1001)
        reference_curves[var] = {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "n_subjects": stacked.shape[0],
        }
        print(f"{var}: averaged across {stacked.shape[0]} subjects, "
              f"mean range [{reference_curves[var]['mean'].min():.2f}, "
              f"{reference_curves[var]['mean'].max():.2f}] degrees")

    return reference_curves


def lookup_reference_value(reference_curves, variable, phase_var):
    """
    Given a phase_var value (0.0 to 1.0, our simulation's live gait-cycle
    position), returns the real reference angle at that point in the
    stride. This is the function our new reward terms will call every step.
    """
    index = int(np.clip(phase_var, 0.0, 1.0) * 1000)
    return reference_curves[variable]["mean"][index]


if __name__ == "__main__":
    print(f"Building reference trajectories from {N_SUBJECTS_TO_USE} real able-bodied subjects...\n")
    reference_curves = build_reference(FILEPATH, VARIABLES_TO_EXTRACT, N_SUBJECTS_TO_USE)

    # Save for reuse in the actual reward function code later
    np.savez("Results/reference_trajectories.npz",
             **{f"{var}_mean": reference_curves[var]["mean"] for var in VARIABLES_TO_EXTRACT},
             **{f"{var}_std": reference_curves[var]["std"] for var in VARIABLES_TO_EXTRACT})
    print("\nSaved reference trajectories to Results/reference_trajectories.npz")

    # Quick sanity check: look up hip angle at a few phase points
    print("\n=== Sanity check: hip angle lookup at various phase_var values ===")
    for phase in [0.0, 0.25, 0.5, 0.75, 1.0]:
        val = lookup_reference_value(reference_curves, "HipAngles", phase)
        print(f"phase_var={phase}: reference hip angle = {val:.2f} degrees")