import numpy as np

real_vectors = {
    "ALS":         np.array([0.142, 1.286, 0.699, 1.257]),
    "Parkinsons":  np.array([0.016, 0.607, 0.568, 0.940]),
    "Huntingtons": np.array([0.427, 0.200, 0.764, 0.428]),
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

diseases = list(real_vectors.keys())
print("=== How similar are the REAL disease directions to each other? ===\n")
for i in range(len(diseases)):
    for j in range(i+1, len(diseases)):
        d1, d2 = diseases[i], diseases[j]
        sim = cosine_sim(real_vectors[d1], real_vectors[d2])
        print(f"{d1} vs {d2}: cosine = {sim:.3f}")