#!/bin/bash

#SBATCH -p mit_normal_gpu                # mit_normal_gpu max walltime is 6 hours
#SBATCH -o output_%j.txt
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 0-06:00
#SBATCH --gres=gpu:2                   # mit_normal_gpu base allocation limit is 2 GPUs per job          # H200 nodes give 15 CPUs/GPU -> 2 GPUs = 30 CPUs
#SBATCH --mem=20GB
#SBATCH --job-name=deprl_myoleg_walk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=milessmi@mit.edu


CONDA_ENV="opensim-env"   # change if deprl lives in a different env
CONDA_ENV_PATH="$HOME/.conda/envs/$CONDA_ENV"
PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
 
module load miniforge
 
# --- Use the env's python binary directly instead of `conda activate` ---
# `conda activate` resolves envs by name via envs_dirs search, which can
# silently miss ~/.conda/envs on shared module installs and fall back to
# base python without erroring. Calling the interpreter by full path
# sidesteps that entirely.
if [ ! -x "$PYTHON_BIN" ]; then
    echo "❌ ERROR: no python found at $PYTHON_BIN"
    echo "   Check that CONDA_ENV_PATH is correct: run 'conda env list' on a login node."
    exit 1
fi
 
echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "
import sys
try:
    import deprl
    print('deprl OK:', deprl.__file__)
except ImportError as e:
    print('MISSING: deprl ->', e)
    sys.exit(1)
try:
    import myosuite
    print('myosuite OK:', myosuite.__version__)
    if myosuite.__version__ != '2.1.5':
        print('WARNING: deprl expects myosuite==2.1.5, found', myosuite.__version__)
except ImportError as e:
    print('MISSING: myosuite ->', e)
    sys.exit(1)
" || {
    echo "❌ ERROR: required package(s) not importable in env '$CONDA_ENV'."
    echo "   On a login node: conda activate $CONDA_ENV && pip install deprl myosuite==2.1.5"
    exit 1
}
 
# --- GPU / threading setup ---

echo "Assigned GPUs: $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}   # falls back to 1 if run outside sbatch
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Headless rendering setup ---
# Compute nodes have no display/X server, so MuJoCo's default GLFW windowed
# context fails with an X11 error even when you never call render(). Force
# an offscreen, GPU-accelerated backend instead. If EGL isn't wired up on
# this node, fall back to the (slower, CPU-only) osmesa backend.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
 
nvidia-smi || echo "⚠️  nvidia-smi not found; check GPU allocation."
 
echo "Environment set up. Starting deprl run..."
 
# ================== Auto-chaining ==================
# deprl auto-resumes from checkpoints, but mit_normal_gpu caps jobs at 6h,
# well short of the 1e8-step training target. Rather than manually
# resubmitting after every timeout, queue the next 6h chunk right now,
# dependent on this job finishing (for any reason, including timeout).
# A safety cap prevents runaway resubmission if something gets stuck.
TARGET_STEPS=100000000
MAX_CHAINS=40   # ~40 * 6h = 240h (~10 days) ceiling before auto-chaining stops on its own
CHAIN_COUNT=${CHAIN_COUNT:-0}
CKPT_BASE="Code/controllers/imitation_reward/training_full_purerep"
 
CKPT_DIR=$(find "$CKPT_BASE" -maxdepth 2 -type d -name checkpoints 2>/dev/null | head -1)
 
CURRENT_STEPS=$("$PYTHON_BIN" -c "
import glob, os, sys
ckpt_dir = '$CKPT_DIR'
if not ckpt_dir or not os.path.isdir(ckpt_dir):
    print(0); sys.exit()
steps = [int(os.path.basename(f).split('.')[0][5:]) for f in glob.glob(os.path.join(ckpt_dir, 'step_*.pt'))]
print(max(steps) if steps else 0)
")
 
echo "Chain iteration: $CHAIN_COUNT | Current steps: $CURRENT_STEPS / $TARGET_STEPS"
 
if [ "$CURRENT_STEPS" -ge "$TARGET_STEPS" ]; then
    echo "✅ Already at/past target step count — not queuing another chained job."
elif [ "$CHAIN_COUNT" -ge "$MAX_CHAINS" ]; then
    echo "⚠️  Reached max chain count ($MAX_CHAINS) without hitting target steps."
    echo "   Stopping auto-resubmission — check the run before manually continuing."
else
    NEXT_CHAIN=$((CHAIN_COUNT + 1))
    echo "Queuing next chained job (iteration $NEXT_CHAIN of $MAX_CHAINS), dependent on job $SLURM_JOB_ID ending..."
    sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,CHAIN_COUNT=$NEXT_CHAIN "$0"
fi
 
"$PYTHON_BIN" -m deprl.main Code/controllers/imitation_reward/myoLegWalkPureReplication_full.json
 
echo "Script completed!"
 
