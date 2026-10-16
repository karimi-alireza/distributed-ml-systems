#!/usr/bin/env bash
#SBATCH --job-name=Q1_MMMC
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV_ACTIVATE=${DMLS_VENV_ACTIVATE:-/home/shared_files/pytorch_venv/bin/activate}
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

export SCRIPT_DIR
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
if [[ -z ${MASTER_PORT:-} ]]; then
    MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
fi
export MASTER_PORT
export OMP_NUM_THREADS=1

# Required accumulated run: physical batch 32, four micro-batches per update.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    cd "$SCRIPT_DIR"
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank="$SLURM_NODEID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --rdzv-id="$SLURM_JOB_ID-accum" \
        Q1_MMMC.py --run-name Q1_MMMC_accum
'

# Question 1.3: keep the physical mini-batch fixed and disable accumulation.
# The resulting effective batch is 32 instead of 128, matching the reference
# experiment and making the cost of synchronizing every mini-batch visible.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    cd "$SCRIPT_DIR"
    torchrun \
        --nnodes=2 \
        --nproc-per-node=2 \
        --node-rank="$SLURM_NODEID" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        --rdzv-id="$SLURM_JOB_ID-no-accum" \
        Q1_MMMC.py \
        --run-name Q1_MMMC_no_accum \
        --batch-size 32 \
        --grad-accum-steps 1
'
