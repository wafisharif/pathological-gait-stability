from myosuite.utils import gym

env = gym.make('myoLegWalk-v0')
env.reset()

print("=== Checking for perturbation-related attributes/methods ===")
unwrapped = env.unwrapped

# Check common naming patterns used across MyoSuite/MuJoCo envs for push/perturbation support
candidates = ["apply_perturbation", "push", "external_force", "perturb", "xfrc_applied"]
for name in candidates:
    has_attr = hasattr(unwrapped, name) or hasattr(unwrapped.sim.data, name)
    print(f"{name}: {'FOUND' if has_attr else 'not found'}")

print("\n=== sim.data.xfrc_applied (raw MuJoCo external force buffer) ===")
try:
    print("Shape:", unwrapped.sim.data.xfrc_applied.shape)
    print("This is MuJoCo's built-in external force buffer -- if it exists, we can apply pushes manually by writing into it, even without a named 'perturbation' API.")
except Exception as e:
    print("Not accessible:", e)

env.close()