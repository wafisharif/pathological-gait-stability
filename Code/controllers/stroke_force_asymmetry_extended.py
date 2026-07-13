from myosuite.utils import gym
import deprl
import numpy as np

def measure_force_asymmetry(policy, paretic_side, paretic_strength, max_steps=1000):
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
    r_forces, l_forces = [], []
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        r = sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0]
        l = sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0]
        r_forces.append(r)
        l_forces.append(l)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    r_forces, l_forces = np.array(r_forces), np.array(l_forces)
    transient = min(30, steps // 4)

    if steps <= transient:
        return steps, float('nan'), float('nan'), float('nan')

    r_post, l_post = r_forces[transient:], l_forces[transient:]
    r_mean, l_mean = np.mean(r_post), np.mean(l_post)
    r_std, l_std = np.std(r_post), np.std(l_post)

    if paretic_side == "left":
        paretic_force, nonparetic_force = l_mean, r_mean
    else:
        paretic_force, nonparetic_force = r_mean, l_mean

    denom = paretic_force + nonparetic_force
    force_asymmetry = (nonparetic_force - paretic_force) / denom if denom > 0 else float('nan')

    return steps, paretic_force, nonparetic_force, force_asymmetry, r_std, l_std


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("EXTENDED FORCE ASYMMETRY SWEEP (up to 1000 steps) — LEFT paretic\n")
    print("paretic_str, steps, paretic_force, nonparetic_force, force_asymmetry, r_std, l_std")
    levels = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    for ps in levels:
        result = measure_force_asymmetry(policy, "left", ps, max_steps=1000)
        steps, pf, nf, fa = result[0], result[1], result[2], result[3]
        r_std, l_std = result[4] if len(result) > 4 else float('nan'), result[5] if len(result) > 5 else float('nan')
        print(f"{ps}, {steps}, {pf:.1f}, {nf:.1f}, {fa:.4f}, {r_std:.1f}, {l_std:.1f}")