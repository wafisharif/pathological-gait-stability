## What this is

This adds a new, real-human-data-based reward to our MyoLeg walking policy's training process, on top of (not replacing) the existing DEP-RL goal-driven reward. This is Phase 1 of the plan Miles approved: improve the healthy baseline's realism first, then re-apply our existing impairment mechanisms on top of the improved baseline and re-benchmark.

## Background

We confirmed by reading the environment's source code (`myosuite/envs/myo/myobase/walk_v0.py`) that the existing reward has no real human reference data anywhere in it -- every term is synthetic (a hand-written cosine wave for hip rhythm, a "don't rotate away from your start pose" term, etc.). This matches what the "Natural and Robust Walking" paper (Schumacher et al., 2023, arXiv:2309.02976) found independently.

We built real, phase-indexed reference curves for hip, knee, ankle, and pelvis angle from 30 real able-bodied subjects in the full-body motion capture dataset (van Criekinge et al., 2023), then measured how our simulation's own joint-angle values relate to those real curves:

| Joint   | Shape correlation (sim vs real)  | Correction applied            | Reward weight |
|---------|----------------------------------|-------------------------------|---------------|
| Hip     | 0.862 (strong)                   | scale 2.04, offset -11.50     | High (3.0)    |
| Pelvis  | stable in both signals           | offset -93.88 (frame mismatch)| High (3.0)    |
| Knee    | 0.435 (weak)                     | scale 2.40, offset +17.86     | Low (0.5)     |
| Ankle   | 0.262 (very weak)                | scale 2.86, offset -25.71     | Low (0.5)     |

Per Miles's approved decision: hip and pelvis get a strong, confident reward weight since their real-vs-sim shape match is trustworthy; knee and ankle get a much lower weight.

## Effort/pain cost function (added after initial imitation reward)

We measured that our simulated muscle activation runs 1.4x-3.2x higher on average than real EMG data (6 leg muscles compared against 30 real able-bodied subjects). Literature research (Schumacher, Geijtenbeek, Caggiano, Kumar, Schmitt, Martius, Haeufle, 2023, "Natural and Robust Walking using RL without Demonstrations in High-Dimensional Musculoskeletal Models," arXiv:2309.02976 -- the direct follow-up to DEP-RL by the same authors, applied to this exact MyoLeg model) explicitly documents this as a known DEP-RL failure mode ("large co-contraction levels"), and provides the exact cost function and coefficients used to fix it:

    c_effort = alpha(t) * a^3 + w1*(u - u_prev)^2 + w2*N_active
    c_pain   = w3 * joint_limit_violations + w4 * excess_GRF

Using their exact reported values (Table IV(d)): w1=0.097, w2=1.579 (15% activation = "active"), w3=0.131, w4=0.073, GRF threshold = 1.2x body weight.

We also directly investigated whether a real EMG-CORRELATION reward term (matching muscle timing directly, the way we did for joint angles) would be worthwhile, and found real, quantitative evidence that it wouldn't reliably work: published state-of-the-art imitation-learning-specific systems (KINESIS, arXiv:2503.14637; MuscleMimic, arXiv:2603.25544) only achieve modest EMG correlations (0-0.6) due to muscle redundancy, and non-imitation baselines like DEP-RL show substantially weaker alignment than that. We deliberately did not build a direct EMG-matching term for this reason.

### Known adaptations from the paper's exact method (documented, not silent)

- The paper's joint-limit pain term uses MuJoCo's internal constraint torque, which we judged too risky to extract without extensive separate verification. We substitute a directly-verifiable proxy: penalizing joint-angle proximity to its known range limit.
- The paper's effort-cubed term uses an ADAPTIVE weight alpha(t) that increases based on training performance relative to a threshold (theta=1000) tuned to their specific reward scale. Our reward combines additional terms at a different scale, so this threshold cannot be safely copied. We use a constant placeholder weight instead. Recalibrating this adaptive schedule to our reward's actual scale (using real training-return data once available) is a well-defined follow-up, not yet done.

### Bugs found and fixed during local verification (documented for transparency)

- `act_mag` (inherited from the original environment) had shape (1,1) instead of a scalar, causing a "setting an array element with a sequence" crash when summed with other terms -- fixed by explicit flattening.
- `active_muscle_count` and `grf_pain` were initially implemented as RAW cumulative counts/forces per episode (unnormalized), causing them to dominate the total reward by roughly 10-20x versus every other term once real walking behavior developed (confirmed via direct per-episode reward-component logging, not assumption). Fixed by normalizing to a 0-1 fraction of muscles (for active count) and to body-weight-relative force (for GRF pain), consistent with how force is scaled elsewhere in this project's perturbation-testing work.
- Both fixes were verified against real, multi-epoch local training runs (via the actual `deprl.main` command, not just isolated environment stepping) before this handoff, confirming stable, bounded episode scores (-270 to +260) across 5 full epochs.

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