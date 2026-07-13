import numpy as np

def load_trace(weakness_factor):
    filename = f"Code/controllers/gait_trace_w{weakness_factor}.npy"
    trace = np.load(filename, allow_pickle=True).item()
    return trace

def summarize_trace(trace, label):
    print(f"\n=== {label} ===")
    print("Number of steps recorded:", trace["qpos_without_xy"].shape[0])

    # Foot heights: shape (steps, 2) -- one column per foot
    feet_heights = trace["feet_heights"]
    print("Mean foot heights (per foot):", np.mean(feet_heights, axis=0))
    print("Std foot heights (per foot):", np.std(feet_heights, axis=0))

    # Joint angles: shape (steps, 33)
    qpos = trace["qpos_without_xy"]
    print("Mean joint angle magnitude (overall):", np.mean(np.abs(qpos)))
    print("Std joint angle magnitude (overall):", np.std(qpos))

    # Phase variable: shape (steps, 1)
    phase = trace["phase_var"].flatten()
    print("Phase var range: min =", np.min(phase), ", max =", np.max(phase))


if __name__ == "__main__":
    levels_to_compare = [1.0, 0.8, 0.6, 0.4, 0.2]

    traces = {}
    for w in levels_to_compare:
        traces[w] = load_trace(w)
        summarize_trace(traces[w], label=f"weakness={w}")