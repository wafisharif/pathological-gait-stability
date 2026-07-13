"""
Given that absolute sim-vs-real distance is confirmed large (3 independent
PCA analyses agree), this checks a more modest, defensible question:
does the RELATIVE ranking of which real disease each combo is "least far"
from make mechanistic sense, even if none are close in absolute terms?
"""
import pandas as pd

df = pd.read_csv("Results/pca_results_v3_disease_fitted.csv")

real = df[df["is_real"] == True]
sim = df[df["is_real"] == False]

real_means = real.groupby("combo_name")[["PC1", "PC2"]].mean()

print("Real disease positions (for reference):")
print(real_means)
print()

print("=== Relative ranking check (not absolute distance) ===")
for combo in sorted(sim["combo_name"].unique()):
    subset = sim[sim["combo_name"] == combo]
    pc1, pc2 = subset["PC1"].mean(), subset["PC2"].mean()

    distances = {}
    for disease in real_means.index:
        r1, r2 = real_means.loc[disease, "PC1"], real_means.loc[disease, "PC2"]
        distances[disease] = ((pc1 - r1)**2 + (pc2 - r2)**2) ** 0.5

    ranked = sorted(distances.items(), key=lambda x: x[1])
    print(f"\n{combo}:")
    for disease, dist in ranked:
        print(f"  {disease}: {dist:.2f}")
    print(f"  Ranking (nearest to farthest): {' < '.join([d for d, _ in ranked])}")