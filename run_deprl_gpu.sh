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
 
nvidia-smi || echo "⚠️  nvidia-smi not found; check GPU allocation."
 
echo "Environment set up. Starting deprl run..."
 
"$PYTHON_BIN" -m deprl.main Code/controllers/imitation_reward/myoLegWalkStrokeTier3_full.json
 
echo "Script completed!"
