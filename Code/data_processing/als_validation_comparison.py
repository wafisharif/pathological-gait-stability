from myosuite.utils import gym
import deprl
import numpy as np

# ============================================================
# REAL GAITNDD REFERENCE VALUES (computed last session)
# Source: PhysioNet Gait Dynamics in Neurodegenerative Disease
# ============================================================
REAL = {
    "Healthy": {
        "mean_stride_time_s":     1.097,
        "double_support_pct":    28.225,
        "left_stance_pct":       63.828,
        "right_stance_pct":      64.394,
    },
    "ALS": {
        "mean_stride_time_s":     1.469,
        "double_support_pct":    40.767,
        "left_stance_pct":       67.688,
        "right_stance_pct":      67.980,
    }
}

# ============================================================
# BEST ALS CONTROLLER CONFIGURATION (locked last session)
# Primary: global strength 0.6x
# Secondary: plantarflexor additional 0.6x
# Secondary: hip extensor additional 0.6x
# ============================================================
PLANTARFLEXOR_NAMES = ['soleus', 'gaslat', 'gasmed']
HIP_EXTENSOR_NAMES  = ['glmax', 'semimem', 'semiten', 'bflh']

BASE_STRENGTH    = 0.6
PF_FACTOR        = 0.6
HIPEXT_FACTOR    = 0.6
CONTACT_THRESHOLD = 10.0


def apply_als_config(sim, base_strength, pf_factor, hipext_factor):
    muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
    pf_indices      = [i for i, n in enumerate(muscle_names)
                       if any(n.startswith(p) for p in PLANTARFLEXOR_NAMES)]
    hipext_indices  = [i for i, n in enumerate(muscle_names)
                       if any(n.startswith(p) for p in HIP_EXTENSOR_NAMES)]

    original = sim.model.actuator_gainprm.copy()
    scaled   = original.copy()
    scaled[:, 2]             = original[:, 2] * base_strength
    scaled[pf_indices, 2]    = original[pf_indices, 2]    * base_strength * pf_factor
    scaled[hipext_indices, 2]= original[hipext_indices, 2] * base_strength * hipext_factor
    sim.model.actuator_gainprm[:] = scaled


def measure_full_gait_profile(policy, label, apply_config_fn=None, max_steps=1000,
                               contact_threshold=CONTACT_THRESHOLD):
    env  = gym.make('myoLegWalk-v0')
    sim  = env.unwrapped.sim

    if apply_config_fn:
        apply_config_fn(sim)

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    r_force_trace, l_force_trace = [], []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        r = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
        l = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]
        r_force_trace.append(r)
        l_force_trace.append(l)
        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    r_arr = np.array(r_force_trace)
    l_arr = np.array(l_force_trace)
    transient = min(30, steps_survived // 4)
    r_post = r_arr[transient:]
    l_post = l_arr[transient:]

    if len(r_post) < 10:
        print(f"  {label}: too few steps to measure")
        return None

    r_on = r_post > contact_threshold
    l_on = l_post > contact_threshold

    # gait phase classification
    both   = r_on & l_on
    only_r = r_on & ~l_on
    only_l = l_on & ~r_on
    n = len(r_on)

    double_support_pct = np.sum(both)   / n * 100
    r_stance_pct       = np.sum(r_on)   / n * 100
    l_stance_pct       = np.sum(l_on)   / n * 100
    r_swing_pct        = (1 - np.sum(r_on)/n) * 100
    l_swing_pct        = (1 - np.sum(l_on)/n) * 100

    # stride timing from footstrike intervals
    def get_strike_intervals(on_signal):
        strikes = [i for i in range(1, len(on_signal))
                   if on_signal[i] and not on_signal[i-1]]
        if len(strikes) < 2:
            return float('nan'), float('nan')
        durations = np.diff(strikes)
        # convert from sim steps to seconds (dt=0.01s per step)
        durations_s = durations * 0.01
        return np.mean(durations_s), np.std(durations_s)

    mean_stride_r, std_stride_r = get_strike_intervals(r_on)
    mean_stride_l, std_stride_l = get_strike_intervals(l_on)
    mean_stride = np.nanmean([mean_stride_r, mean_stride_l])

    distance = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    speed    = distance / max(steps_survived, 1)

    n_strikes = (np.sum((r_on[1:]) & (~r_on[:-1])) +
                 np.sum((l_on[1:]) & (~l_on[:-1])))

    return {
        "label":                label,
        "steps_survived":       steps_survived,
        "n_strikes":            int(n_strikes),
        "distance_m":           distance,
        "mean_stride_time_s":   mean_stride,
        "double_support_pct":   double_support_pct,
        "right_stance_pct":     r_stance_pct,
        "left_stance_pct":      l_stance_pct,
        "right_swing_pct":      r_swing_pct,
        "left_swing_pct":       l_swing_pct,
    }


def print_comparison(sim_healthy, sim_als):
    metrics = [
        ("mean_stride_time_s",  "Stride time (s)",      "Higher = slower gait"),
        ("double_support_pct",  "Double-support %",     "Higher = more time both feet down"),
        ("right_stance_pct",    "Right stance %",       "Higher = more time on right foot"),
        ("left_stance_pct",     "Left stance %",        "Higher = more time on left foot"),
    ]

    print("\n" + "="*80)
    print("ALS CONTROLLER VALIDATION — SIMULATED vs REAL GaitNDD")
    print("="*80)
    print(f"{'Metric':<25} {'Real Healthy':>14} {'Real ALS':>12} {'Sim Healthy':>13} {'Sim ALS':>10} {'Direction OK?':>14}")
    print("-"*80)

    for key, label, note in metrics:
        rh  = REAL["Healthy"].get(key, float('nan'))
        ra  = REAL["ALS"].get(key, float('nan'))
        sh  = sim_healthy[key] if sim_healthy else float('nan')
        sa  = sim_als[key]     if sim_als     else float('nan')

        real_direction = "↑" if ra > rh else "↓"
        sim_direction  = "↑" if sa > sh else "↓"
        match = "✅" if real_direction == sim_direction else "❌"

        print(f"{label:<25} {rh:>14.2f} {ra:>12.2f} {sh:>13.2f} {sa:>10.2f} {match:>10} {real_direction}→{sim_direction}")

    print("-"*80)
    print(f"\nSimulated healthy: {sim_healthy['steps_survived']} steps, {sim_healthy['distance_m']:.2f}m, {sim_healthy['n_strikes']} strikes")
    print(f"Simulated ALS:     {sim_als['steps_survived']} steps, {sim_als['distance_m']:.2f}m, {sim_als['n_strikes']} strikes")
    print()


if __name__ == "__main__":
    # load policy once
    env    = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("Running simulated HEALTHY baseline (0.85x strength -- full-strength falls too fast for reliable measurement)...")
    sim_healthy = measure_full_gait_profile(
        policy, label="Sim Healthy (0.85x)",
        apply_config_fn=lambda sim: apply_als_config(
            sim, 0.85, 1.0, 1.0   # only global strength, no plantarflexor or hip extensor changes
        )
    )

    print("Running simulated ALS controller (Version 1)...")
    sim_als = measure_full_gait_profile(
        policy, label="Sim ALS (v1)",
        apply_config_fn=lambda sim: apply_als_config(
            sim, BASE_STRENGTH, PF_FACTOR, HIPEXT_FACTOR
        )
    )

    if sim_healthy and sim_als:
        print_comparison(sim_healthy, sim_als)