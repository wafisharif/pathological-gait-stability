from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

# Use our known-good, full-survival strength level (0.85) for this check
original_gainprm = sim.model.actuator_gainprm.copy()
scaled = original_gainprm.copy()
scaled[:, 2] = original_gainprm[:, 2] * 0.85
sim.model.actuator_gainprm[:] = scaled

obs, _ = env.reset()
r_force, l_force = [], []

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r_force.append(sim.data.sensor('r_foot').data[0] + sim.data.sensor('r_toes').data[0])
    l_force.append(sim.data.sensor('l_foot').data[0] + sim.data.sensor('l_toes').data[0])
    if terminated or truncated:
        break

env.close()
r_force = np.array(r_force[30:])
l_force = np.array(l_force[30:])

print("Right foot force: min={:.2f}, max={:.2f}, mean={:.2f}".format(r_force.min(), r_force.max(), r_force.mean()))
print("Left foot force: min={:.2f}, max={:.2f}, mean={:.2f}".format(l_force.min(), l_force.max(), l_force.mean()))

print("\nNonzero (>0) force distribution (right foot), percentiles:")
nonzero_r = r_force[r_force > 0]
if len(nonzero_r) > 0:
    for p in [5, 10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(nonzero_r, p):.2f}")

print("\nFor various thresholds, % time BOTH feet register contact, % NEITHER foot registers contact:")
for thresh in [1, 5, 10, 20, 50, 100]:
    r_on = r_force > thresh
    l_on = l_force > thresh
    both = np.mean(r_on & l_on) * 100
    neither = np.mean(~r_on & ~l_on) * 100
    either = np.mean(r_on | l_on) * 100
    print(f"  threshold={thresh}: both={both:.1f}%, neither={neither:.1f}%, at_least_one={either:.1f}%")