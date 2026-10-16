# CA4 - CPU Distributed Training and Profiling

This directory implements all code deliverables requested by `DMLS_CA4.pdf`.
The training data is read directly from the supplied ZIP files, so extraction is
optional. Feature arrays stay as `float16` in memory and are converted to
`float32` one mini-batch at a time.

## Files

- `Q1_*.py` and `Q1_*.sh`: standalone manual PyTorch DDP solutions with the
  CPU-compatible Gloo backend, `torchrun`, Slurm, gradient accumulation, and
  `no_sync()` on intermediate micro-batches.
- `Q2_*.py` and `Q2_*.sh`: standalone versions of the same experiment using
  Hugging Face Accelerate and `accelerator.accumulate()`. The Slurm scripts use
  `torchrun` to create the CPU workers; Accelerate detects the torchrun rank
  environment and manages the distributed training inside each Python process.
- `Profiler.ipynb`: CPU profiling of 100 test batches and comparison of ReLU,
  Sigmoid, Tanh, and GeLU.
- `ca4_common.py`: the model/data helper used only by the profiling notebook.
- `REPORT.md`: a structured report template for the cluster and Colab results.

The filename abbreviations follow the handout:

| Name | Machines | Processes per machine | Total processes |
|---|---:|---:|---:|
| SMSC | 1 | 1 | 1 |
| SMMC | 1 | 2 | 2 |
| MMSC | 2 | 1 | 2 |
| MMMC | 2 | 2 | 4 |

## Running on the course cluster

Submit each configuration from this directory:

```bash
sbatch Q1_SMSC.sh
sbatch Q1_SMMC.sh
sbatch Q1_MMSC.sh
sbatch Q1_MMMC.sh

sbatch Q2_SMSC.sh
sbatch Q2_SMMC.sh
sbatch Q2_MMSC.sh
sbatch Q2_MMMC.sh
```

The scripts use `/home/shared_files/pytorch_venv/bin/activate` when it exists.
Override it if the cluster path differs:

```bash
DMLS_VENV_ACTIVATE=/actual/path/bin/activate sbatch Q2_SMSC.sh
```

Each run writes a rank-zero-only checkpoint to `checkpoints/` and a compact JSON
record to `results/`. `Q1_MMMC.sh` also performs the required no-accumulation
comparison using the same physical batch size (32) and one backward pass per
optimizer update.

For a quick single-process smoke run on a machine with PyTorch installed:

```bash
torchrun --standalone --nproc-per-node=1 Q1_SMSC.py --epochs 1
torchrun --standalone --nproc-per-node=1 Q2_SMSC.py --epochs 1
```

## Profiling

Question 3 must run on Colab according to the handout. Upload this directory (or
mount it from Drive), open `Profiler.ipynb`, and run all cells. The notebook does
not train the model; it profiles the same architecture on exactly 100 test
batches for each activation function.

## Submission

Include the eight named Python scripts, eight named shell scripts,
`Profiler.ipynb`, `ca4_common.py`, and the final PDF report. Each named training
script is self-contained. The data archives, checkpoints, result JSON files,
Slurm logs, and assignment PDF do not need to be submitted unless the instructor
asks for them.
