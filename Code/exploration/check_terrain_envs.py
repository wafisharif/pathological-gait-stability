from myosuite.utils import gym
import deprl

terrain_envs_to_try = [
    'myoLegWalk-v0',                # flat, our existing baseline for comparison
    'myoLegRoughTerrainWalk-v0',
    'myoLegHillyTerrainWalk-v0',
    'myoLegStairTerrainWalk-v0',
]

for env_name in terrain_envs_to_try:
    print(f"\n=== Trying {env_name} ===")
    try:
        env = gym.make(env_name)
        obs, _ = env.reset()
        print(f"  Loaded successfully. Obs shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")

        # Try loading our existing flat-terrain baseline policy on this env
        policy = deprl.load_baseline(env)
        print("  Baseline policy loaded onto this env without error.")

        # Quick 50-step test run
        steps_survived = 0
        for step in range(50):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_survived += 1
            if terminated or truncated:
                break
        print(f"  Survived {steps_survived}/50 test steps.")

        env.close()
    except Exception as e:
        print(f"  FAILED: {e}")