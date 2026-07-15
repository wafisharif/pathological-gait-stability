## Reward Function — Phase 1

## Description

This replaces the reward our MyoLeg walking policy trains against with a real, published formula, instead of the made-up one it had before. It's the first real step in the plan Miles approved: get the healthy baseline walking more realistically, then put our existing impairment mechanisms back on top of it and see if that finally fixes the problems we've been running into.

## Background

I went and read the environment's source code (`myosuite/envs/myo/myobase/walk_v0.py`) and confirmed that the existing reward has no real human data in it anywhere. It's rewarding the walker for things like following a hand-written cosine wave for hip motion, or just staying close to whatever position it started in. None of it is related to how a real person actually walks.

So I swapped it out for the exact reward formula used in "Natural and Robust Walking using RL without Demonstrations in High-Dimensional Musculoskeletal Models". Their version achieved a 43% (± 5%) match against actual human movement data, which is as strong of a track record as I found for this problem.

The formula is:

    r = r_vel - c_effort - c_pain
    c_effort = alpha(t) * a^3 + w1*(u - u_prev)^2 + w2*N_active
    c_pain   = w3 * joint_limit_violations + w4 * excess_GRF

I used their reported coefficients: w1=0.097, w2=1.579 (with 15% activation counting as "active"), w3=0.131, w4=0.073, and a ground-reaction-force threshold of 1.2 times body weight. The effort term uses their real adaptive weight schedule too (Algorithm 1 in the paper, with delta_alpha=9e-4, theta=1000, beta=0.8, and lambda=0.9), which starts the effort penalty at zero and only ramps it up once the policy is actually walking well.

## One thing I adapted

The paper's joint-limit pain term uses MuJoCo's internal constraint torque, and pulling that out felt risky to do without more testing. So instead I built a proxy that penalizes a joint for getting close to its range limit, which captures the same idea.

## What I tested

I ran this through the real training command locally, and it works. Every debug run I did had episodes run for the full 1000 steps without falling, across multiple epochs. I also checked the adaptive effort schedule and confirmed it's behaving as it should: it sits at zero while the policy hasn't learned to walk yet, since its average return is nowhere near the threshold, and it's only supposed to start climbing once that changes. On my Mac, with no dedicated GPU, I'm getting between 15 and 45 steps per second depending on config, so a 100-million-step run would take way too long.

## Folder contents

`myoleg_walk_pure_replication.py` is the environment class. `myoLegWalkPureReplication_debug.json` is the config I used for testing. `myoLegWalkPureReplication_full.json` is the file to run, set up for the full 100-million-step run across 10 parallel environments.

## How to run

    python -m deprl.main Code/controllers/imitation_reward/myoLegWalkPureReplication_full.json

Everything it saves, like checkpoints and logs and everything else, will go to `Code/controllers/imitation_reward/training_full_purerep/`.

## Once it's done training

Please send back the checkpoint and I'll continue from there. I'll load it the same way I load the current baseline, run it through the benchmarking scripts, and then compare it against GaitNDD and stroke mocap data. If it looks better, I'll put the existing impairment controllers back on top of it and re-run validation.