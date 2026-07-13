## What this is

This adds a new, real-human-data-based reward to our MyoLeg walking policy's training process, on top of (not replacing) the existing DEP-RL goal-driven reward. This is Phase 1 of the plan Miles approved: improve the healthy baseline's realism first, then re-apply our existing impairment mechanisms on top of the improved baseline and re-benchmark.

## Background

We confirmed by reading the environment's source code (`myosuite/envs/myo/myobase/walk_v0.py`) that the existing reward has no real human reference data anywhere in it -- every term is synthetic (a hand-written cosine wave for hip rhythm, a "don't rotate away from your start pose" term, etc.). This matches what the "Natural and Robust Walking" paper (Schumacher et al., 2023) found independently.

We built real, phase-indexed reference curves for hip, knee, ankle, and pelvis angle from 30 real able-bodied subjects in the full-body motion capture dataset (van Criekinge et al., 2023), then measured how our simulation's own joint-angle values relate to those real curves:

| Joint   | Shape correlation (sim vs real)  | Correction applied            | Reward weight |
|---------|----------------------------------|-------------------------------|---------------|
| Hip     | 0.862 (strong)                   | scale 2.04, offset -11.50     | High (3.0)    |
| Pelvis  | stable in both signals           | offset -93.88 (frame mismatch)| High (3.0)    |
| Knee    | 0.435 (weak)                     | scale 2.40, offset +17.86     | Low (0.5)     |
| Ankle   | 0.262 (very weak)                | scale 2.86, offset -25.71     | Low (0.5)     |

Hip and pelvis get a strong, confident reward weight since their real-vs-sim shape match is trustworthy; knee and ankle get a much lower weight, since forcing them toward a poorly-correlated target risks fighting the training process rather than helping it.

## What's in this folder

- `myoleg_walk_imitation_env.py` — the new environment class (`WalkEnvV0Imitation`), a subclass of MyoSuite's `WalkEnvV0` that adds four new reward terms (`hip_imitation`, `pelvis_imitation`, `knee_imitation`, `ankle_imitation`) alongside all existing reward terms.
- `reference_trajectories.npz` — the real, averaged reference curves this environment loads at startup.
- `myoLegWalkImitation_debug.json` — small-scale config (2,000 steps, 2 parallel environments) used for local testing only. Already run successfully on a MacBook (no GPU) -- confirms the code runs correctly end-to-end, completes training epochs, saves checkpoints, and shuts down cleanly.
- `myoLegWalkImitation_full.json` — the real, full-scale config (100 million steps, 10 parallel environments), matching the exact scale of the original baseline's training config. **This is the one to actually run for real training.**

## What's already been verified locally (no GPU needed for this part)

- Environment registers and runs correctly (confirmed via direct testing)
- Full training loop runs via the real `deprl.main` command, not just a standalone script -- confirmed two full epochs complete, checkpoints save correctly, and all four new reward terms log correctly alongside the existing ones
- At local (CPU/MPS, no dedicated GPU) speed: ~38-39 steps/second. At that rate, the full 100-million-step run would take an estimated 25-30+ days locally -- this is why it needs to run on real GPU hardware.

## How to run the real training

```bash
python -m deprl.main Code/controllers/imitation_reward/myoLegWalkImitation_full.json
```

Output (checkpoints, logs, config) will be saved to `Code/controllers/imitation_reward/training_full/`.

## What to do once training finishes

Load the resulting checkpoint the same way we load the current baseline (`deprl.load(path, env)`, using the new `myoLegWalkImitation-v0` environment ID), then run it through our existing benchmarking pipeline (`Code/benchmarking/benchmark_controller.py`) to compare it against real GaitNDD data, the same way we've validated every other controller in
this project. If the new baseline is meaningfully more human-like, the next step is re-applying our existing impairment mechanisms (ALS/stroke/Parkinson's) on top of it and re-running validation -- per the locked Phase 1 plan.

## Known minor issue (not blocking)

Running with `parallel=1` triggers a harmless cleanup error at the very end of training (`'Sequential' object has no attribute 'processes'`), which happens only after all checkpoints have already saved successfully. Using `parallel=2` or higher (as in both configs here) avoids this entirely.