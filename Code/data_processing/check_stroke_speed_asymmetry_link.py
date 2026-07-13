"""
Checks whether real stroke patients show a relationship between walking
speed and asymmetry (analogous to the speed-variability relationship
found for ALS/Parkinson's), to give Combo B's correlation something
real to compare against.
"""
import pandas as pd

stroke_features = pd.read_csv("Results/stroke_mocap_asymmetry_features.csv")

# We don't have direct per-subject walking speed in our stroke asymmetry
# extraction yet -- check what's actually available first
print("Available columns:")
print(list(stroke_features.columns))
print(f"\nNumber of subjects: {len(stroke_features)}")