#!/usr/bin/env bash
#SBATCH --job-name=Q1_SMMC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

VENV_ACTIVATE=${DMLS_VENV_ACTIVATE:-/home/shared_files/pytorch_venv/bin/activate}
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc-per-node=2 Q1_SMMC.py "$@"
