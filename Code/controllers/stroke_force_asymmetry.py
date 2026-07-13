from myosuite.utils import gym
import deprl
import numpy as np

def measure_force_asymmetry(policy, paretic_side, paretic_strength, max_steps=200):
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
    transient = min(20, steps // 4)
    r_mean = np.mean(r_forces[transient:]) if steps > transient else float('nan')
    l_mean = np.mean(l_forces[transient:]) if steps > transient else float('nan')

    if paretic_side == "left":
        paretic_force, nonparetic_force = l_mean, r_mean
    else:
        paretic_force, nonparetic_force = r_mean, l_mean

    denom = paretic_force + nonparetic_force
    force_asymmetry = (nonparetic_force - paretic_force) / denom if denom > 0 else float('nan')

    return steps, paretic_force, nonparetic_force, force_asymmetry


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)
    env.close()

    print("FORCE ASYMMETRY (instantaneous, doesn't require full gait cycles) — LEFT paretic\n")
    print("paretic_str, steps, paretic_force, nonparetic_force, force_asymmetry")
    for ps in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        steps, pf, nf, fa = measure_force_asymmetry(policy, "left", ps)
        print(f"{ps}, {steps}, {pf:.1f}, {nf:.1f}, {fa:.4f}")