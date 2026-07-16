## New Reward Function

## Description

This replaces the reward our walking policy trains against with a published formula.

## Background

I read the environment's source code (`myosuite/envs/myo/myobase/walk_v0.py`) and found that the reward has no human data in it. It's rewarding the walker for things like following a cosine wave for hip motion, or just staying close to whatever position it started in. None of it is related to how a real person walks.

So I swapped it out for the reward formula used in "Natural and Robust Walking using RL without Demonstrations in High-Dimensional Musculoskeletal Models", which resulted in a 43% match against real movement data, which is the strongest match I found.

Formula:

    r = r_vel - c_effort - c_pain
    c_effort = alpha(t) * a^3 + w1*(u - u_prev)^2 + w2*N_active
    c_pain   = w3 * joint_limit_violations + w4 * excess_GRF

I used their coefficients: w1 = 0.097, w2 = 1.579, w3 = 0.131, w4 = 0.073, and a GRF threshold of 1.2x body weight. The effort term uses their real adaptive weight schedule (Algorithm 1 in the paper), which starts effort penalty at zero and increases it when the policy walks well.

## Something I adapted

Their joint-limit pain term uses MuJoCo's constraint torque, but using that felt risky so instead I made a proxy that penalizes joints for getting close to their range limit.

## What I tested

I ran this locally and it works. Every debug I did had episodes go for the full 1000 steps without falling. I also checked the adaptive effort schedule, and that works nicely as well. But on my Mac, with no dedicated GPU, I'm getting between 15 and 45 steps per second, so a 100-million-step run would take way too long.

## Folder contents

`myoleg_walk_pure_replication.py` is the environment class. `myoLegWalkPureReplication_debug.json` is the config I used for testing. `myoLegWalkPureReplication_full.json` is the file to run, set up for the 100-million-step run across 10 environments.

## Run command

    python -m deprl.main Code/controllers/imitation_reward/myoLegWalkPureReplication_full.json

Everything it saves, like checkpoints and logs and everything else, will go to `Code/controllers/imitation_reward/training_full_purerep/`.

## Once it's done training

Please send back the checkpoint and I'll continue from there. I'll load it how I load the current baseline, run it through benchmarking scripts, and then compare it against mocap data. If it looks better, I'll put my impairment controllers on top of it and redo validation.
