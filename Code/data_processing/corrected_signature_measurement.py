from myosuite.utils import gym
import deprl
import numpy as np


def measure_real_signatures(policy, strength_factor, vmax_factor=1.0, max_steps=1000, contact_threshold=10.0):
    """
    Uses REAL contact sensors (foot + toes combined, per leg), with
    CORRECTED labels confirmed via direct simultaneous measurement:
    -- 'left' signals (l_foot + l_toes) = true left foot
    -- 'right' signals (r_foot + r_toes) = true right foot
    (Note: this is independent of feet_heights index ordering, which we
    separately confirmed was backwards from our original assumption.)
    """
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * strength_factor
    scaled[:, 6] = original_gainprm[:, 6] * vmax_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    start_x, start_y = sim.data.qpos[0], sim.data.qpos[1]

    r_contact_force, l_contact_force = [], []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        r_force = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
        l_force = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]
        r_contact_force.append(r_force)
        l_contact_force.append(l_force)

        steps_survived += 1
        if terminated or truncated:
            break

    end_x, end_y = sim.data.qpos[0], sim.data.qpos[1]
    env.close()

    r_contact_force = np.array(r_contact_force)
    l_contact_force = np.array(l_contact_force)

    transient = min(30, steps_survived // 4)
    r_post = r_contact_force[transient:]
    l_post = l_contact_force[transient:]

    if len(r_post) < 5:
        return {"steps_survived": steps_survived, "confidence": "too_short"}

    r_on = r_post > contact_threshold
    l_on = l_post > contact_threshold

    both_on = r_on & l_on
    only_r = r_on & ~l_on
    only_l = l_on & ~r_on
    neither = ~r_on & ~l_on

    n = len(r_on)
    double_support_pct = np.sum(both_on) / n * 100
    single_support_pct = (np.sum(only_r) + np.sum(only_l)) / n * 100
    flight_pct = np.sum(neither) / n * 100  # neither foot down -- shouldn't happen much in normal walking

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # footstrikes from rising edges of contact
    r_strikes = np.sum((r_on[1:]) & (~r_on[:-1]))
    l_strikes = np.sum((l_on[1:]) & (~l_on[:-1]))
    n_strikes = r_strikes + l_strikes
    cadence = n_strikes / n * 1000

    return {
        "steps_survived": steps_survived, "confidence": "ok" if n_strikes >= 4 else "low_confidence",
        "distance": distance, "double_support_pct": double_support_pct,
        "single_support_pct": single_support_pct, "flight_pct": flight_pct,
        "cadence_per_1000steps": cadence, "n_total_strikes": n_strikes,
    }


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("Re-measuring with REAL contact sensors instead of height heuristic\n")
    strength_levels = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]
    print("strength, steps, conf, distance, ds_pct, ss_pct, flight_pct, cadence, n_strikes")
    for s in strength_levels:
        r = measure_real_signatures(policy, strength_factor=s)
        if r.get("confidence") == "too_short":
            print(f"{s}: too short to measure")
            continue
        print(f"{s}, {r['steps_survived']}, {r['confidence']}, {r['distance']:.2f}, "
              f"{r['double_support_pct']:.1f}, {r['single_support_pct']:.1f}, {r['flight_pct']:.1f}, "
              f"{r['cadence_per_1000steps']:.2f}, {r['n_total_strikes']}")