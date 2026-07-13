from myosuite.utils import gym
import deprl
import numpy as np

env = gym.make('myoLegWalk-v0')
sim = env.unwrapped.sim
policy = deprl.load_baseline(env)

obs, _ = env.reset()

records = []
for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    obs_dict = env.unwrapped.get_obs_dict(sim)

    height_idx0 = float(np.array(obs_dict["feet_heights"]).flatten()[0])
    height_idx1 = float(np.array(obs_dict["feet_heights"]).flatten()[1])
    sensor_r = sim.data.sensor('r_foot').data[0]
    sensor_l = sim.data.sensor('l_foot').data[0]

    records.append((step, height_idx0, height_idx1, sensor_r, sensor_l))

    if terminated or truncated:
        print(f"Terminated at step {step}")
        break

env.close()

print(f"\n{'step':>5} {'h_idx0':>8} {'h_idx1':>8} {'sens_r':>10} {'sens_l':>10}")
for r in records[::10]:  # every 10th step for readability
    print(f"{r[0]:>5} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>10.2f} {r[4]:>10.2f}")

# Direct correlation check: when sensor_r is nonzero (real right-foot contact),
# which height index tends to be LOW at that same moment?
records_arr = np.array(records)
r_contact_mask = records_arr[:, 3] > 0  # moments with real right-foot force

if np.sum(r_contact_mask) > 0:
    mean_h0_during_r_contact = np.mean(records_arr[r_contact_mask, 1])
    mean_h1_during_r_contact = np.mean(records_arr[r_contact_mask, 2])
    print(f"\nDuring REAL right-foot contact (sensor_r > 0):")
    print(f"  mean height_idx0 = {mean_h0_during_r_contact:.4f}")
    print(f"  mean height_idx1 = {mean_h1_during_r_contact:.4f}")
    print(f"  (whichever is LOWER during real right contact = that index is the right foot)")
else:
    print("\nNo right-foot contact detected at all in this episode.")