from myosuite.utils import gym
import deprl
import numpy as np

PLANTARFLEXOR_NAMES = ['soleus', 'gaslat', 'gasmed']
HIP_EXTENSOR_NAMES = ['glmax', 'semimem', 'semiten', 'bflh']

def measure(policy, base_strength, pf_factor, hipext_factor, max_steps=1000, contact_threshold=10.0):
    env = gym.make('myoLegWalk-v0')
    sim = env.unwrapped.sim
    muscle_names = [sim.model.actuator(i).name for i in range(sim.model.nu)]
    pf_indices = [i for i, n in enumerate(muscle_names) if any(n.startswith(p) for p in PLANTARFLEXOR_NAMES)]
    hip_indices = [i for i, n in enumerate(muscle_names) if any(n.startswith(p) for p in HIP_EXTENSOR_NAMES)]

    original_gainprm = sim.model.actuator_gainprm.copy()
    scaled = original_gainprm.copy()
    scaled[:, 2] = original_gainprm[:, 2] * base_strength
    scaled[pf_indices, 2] = original_gainprm[pf_indices, 2] * base_strength * pf_factor
    scaled[hip_indices, 2] = original_gainprm[hip_indices, 2] * base_strength * hipext_factor
    sim.model.actuator_gainprm[:] = scaled

    obs, _ = env.reset()
    r_force, l_force = [], []
    steps_survived = 0

    for step in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        r_force.append(sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0])
        l_force.append(sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0])
        steps_survived += 1
        if terminated or truncated:
            break

    env.close()
    r_force, l_force = np.array(r_force), np.array(l_force)
    transient = min(30, steps_survived // 4)
    r_post, l_post = r_force[transient:], l_force[transient:]
    if len(r_post) < 5:
        return steps_survived, "too_short", float('nan'), 0

    r_on, l_on = r_post > contact_threshold, l_post > contact_threshold
    both_on = r_on & l_on
    n_strikes = np.sum((r_on[1:]) & (~r_on[:-1])) + np.sum((l_on[1:]) & (~l_on[:-1]))
    ds_pct = np.sum(both_on) / len(r_on) * 100
    confidence = "ok" if n_strikes >= 6 else "low_confidence"
    return steps_survived, confidence, ds_pct, n_strikes


if __name__ == "__main__":
    env = gym.make('myoLegWalk-v0')
    policy = deprl.load_baseline(env)

    muscle_names = [env.unwrapped.sim.model.actuator(i).name for i in range(env.unwrapped.sim.model.nu)]
    hip_found = [n for n in muscle_names if any(n.startswith(p) for p in HIP_EXTENSOR_NAMES)]
    print(f"Hip extensor muscles found: {hip_found}\n")
    env.close()

    print("Combining plantarflexor (fixed at 0.6, our best so far) + hip extensor weakening, base strength=0.6\n")
    hipext_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    print("hipext_factor, steps, conf, ds_pct, n_strikes")
    for h in hipext_factors:
        steps, conf, ds, n = measure(policy, base_strength=0.6, pf_factor=0.6, hipext_factor=h)
        print(f"{h}, {steps}, {conf}, {ds:.2f}, {n}")