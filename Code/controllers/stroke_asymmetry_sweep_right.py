from myosuite.utils import gym
import deprl
import numpy as np

CONTACT_THRESHOLD = 10.0

def measure_stroke(policy, paretic_side, paretic_strength, max_steps=1000):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original = sim.model.actuator_gainprm.copy()
    scaled = original.copy()
    if paretic_side == "left":
        scaled[40:80, 2] = original[40:80, 2] * paretic_strength
    else:
        scaled[0:40, 2] = original[0:40, 2] * paretic_strength
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]
    r_trace, l_trace = [], []
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        r_trace.append(sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0])
        l_trace.append(sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0])
        steps += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    r = np.array(r_trace); l = np.array(l_trace)
    transient = min(30, steps // 4)
    r, l = r[transient:], l[transient:]
    n = len(r)
    if n < 10:
        return None

    r_on = r > CONTACT_THRESHOLD
    l_on = l > CONTACT_THRESHOLD
    r_stance_pct = np.sum(r_on) / n * 100
    l_stance_pct = np.sum(l_on) / n * 100

    if paretic_side == "left":
        paretic_stance, nonparetic_stance = l_stance_pct, r_stance_pct
    else:
        paretic_stance, nonparetic_stance = r_stance_pct, l_stance_pct

    denom = nonparetic_stance + paretic_stance
    asymmetry_index = (nonparetic_stance - paretic_stance) / denom if denom > 0 else float('nan')

    n_strikes = (np.sum((r_on[1:]) & (~r_on[:-1])) + np.sum((l_on[1:]) & (~l_on[:-1])))
    distance = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    confidence = "ok" if n_strikes >= 6 else "low_conf"

    return {
        "paretic_strength": paretic_strength, "steps": steps, "n_strikes": int(n_strikes),
        "distance": round(distance, 2), "paretic_stance_pct": round(paretic_stance, 2),
        "nonparetic_stance_pct": round(nonparetic_stance, 2),
        "asymmetry_index": round(asymmetry_index, 4), "confidence": confidence,
    }


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("STROKE ASYMMETRY SWEEP — weakening RIGHT leg progressively\n")
    print("paretic_str, steps, conf, paretic_stance%, nonparetic_stance%, asymmetry_index, n_strikes, dist")
    for ps in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        r = measure_stroke(policy, paretic_side="right", paretic_strength=ps)
        if r is None:
            print(f"{ps}: too short to measure")
            continue
        print(f"{r['paretic_strength']}, {r['steps']}, {r['confidence']}, "
              f"{r['paretic_stance_pct']}, {r['nonparetic_stance_pct']}, "
              f"{r['asymmetry_index']}, {r['n_strikes']}, {r['distance']}")