import pandas as pd

filepath = "Results/timing_noise_combo_full.csv"

df = pd.read_csv(filepath)

old_name = "combo_cd_timing_noise"
new_name = "combo_c_feedback_delay_noise"

n_changed = (df["combo_name"] == old_name).sum()
df["combo_name"] = df["combo_name"].replace(old_name, new_name)

df.to_csv(filepath, index=False)

print(f"Relabeled {n_changed} rows from '{old_name}' to '{new_name}'")
print(f"Saved back to {filepath}")