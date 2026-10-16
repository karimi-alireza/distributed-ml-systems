# DMLS Computer Assignment 4 Report

Student ID: 810101492
Date: _fill after experiments_

## 1. Distributed training with gradient accumulation

### 1.1 Model and implementation

The classifier accepts 512-dimensional input vectors. It first normalizes the
input, then applies two hidden blocks of Linear, BatchNorm, and ReLU layers with
256 and 128 hidden units. A final Linear layer produces logits for the 20
classes. Training uses cross-entropy loss and AdamW with learning rate 0.001 and
weight decay 0.0001. A fixed seed of 1404 is used. The feature arrays remain in
their supplied `float16` representation in memory and each mini-batch is
converted to `float32` immediately before its forward pass.

Each process uses a physical mini-batch of 32 and accumulates gradients over
four mini-batches, producing an effective batch of 128 per process. The loss is
divided by the number of mini-batches in the current accumulation group before
backpropagation, including the final partial group. For distributed runs, the
first three forward/backward passes execute inside
`DistributedDataParallel.no_sync()`. Therefore their gradient all-reduces are
skipped and only the fourth backward pass synchronizes the accumulated
gradients before the optimizer update. Synchronizing every backward pass is not
necessary and would add three avoidable communication operations per update.

The Gloo backend is used because all experiments are CPU-only. A
`DistributedSampler` gives each rank a separate training shard, its epoch is
updated before every pass, and barriers surround timing and evaluation. The
maximum elapsed time across ranks is reported. Only rank zero evaluates the
unwrapped model, saves the checkpoint, and writes the JSON result.

### 1.2 Performance results

| Configuration | Machines | Processes/machine | Training time (s) | Speedup | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Q1_SMSC | 1 | 1 | TODO | 1.00 | TODO |
| Q1_SMMC | 1 | 2 | TODO | TODO | TODO |
| Q1_MMSC | 2 | 1 | TODO | TODO | TODO |
| Q1_MMMC_accum | 2 | 2 | TODO | TODO | TODO |

Speedup is calculated as the Q1_SMSC training time divided by the configuration
training time. After inserting the cluster results, compare the two-process
single-machine and two-machine runs to separate local multiprocessing overhead
from network communication. Also state whether every measured model exceeded
the required 80% test accuracy.

### 1.3 Comparison without gradient accumulation

| Mode (2 machines, 2 processes each) | Physical batch/process | Accumulation steps | Global effective batch | Training time (s) | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Accumulation | 32 | 4 | 512 | TODO | TODO |
| No accumulation | 32 | 1 | 128 | TODO | TODO |

The no-accumulation run keeps the physical mini-batch fixed at 32 and performs
an optimizer step and gradient synchronization after every mini-batch. The
accumulated run communicates once every four mini-batches, so it should reduce
communication frequency and optimizer-step overhead. The two runs do not have
the same effective global batch; this is intentional because the experiment
removes accumulation without increasing the physical memory requirement.
Consequently, differences in accuracy can result from both the update frequency
and the effective batch. BatchNorm still observes physical batches of 32 in
both runs.

## 2. Training with Hugging Face Accelerate

### 2.1 Implementation

Question 2 uses the same model, data, optimizer, seed, batch size, accumulation
count, and number of epochs as Question 1. `Accelerator.prepare()` wraps the
model and optimizer and shards both data loaders. The
`accelerator.accumulate(model)` context controls loss scaling, optimizer-step
gating, and DDP gradient synchronization. Predictions and labels are combined
with `gather_for_metrics()`, and only the main process saves the unwrapped state
dictionary and JSON result.

The shell scripts use `torchrun` only to create the requested CPU worker
processes. Once started, Accelerator discovers the standard rank environment
and manages the distributed training. This avoids the single-process behavior
of the `accelerate launch --cpu` simple launcher while preserving Accelerate's
training abstraction. Compared with Question 1, the Python code no longer
constructs a process group or distributed sampler and does not manually select
`no_sync()` contexts. It retains explicit barriers around timing and calls
`end_training()` for orderly process-group cleanup.

### 2.2 Results and comparison

| Configuration | Q1 DDP time (s) | Q2 Accelerate time (s) | Q1 accuracy | Q2 accuracy |
|---|---:|---:|---:|---:|
| SMSC | TODO | TODO | TODO | TODO |
| SMMC | TODO | TODO | TODO | TODO |
| MMSC | TODO | TODO | TODO | TODO |
| MMMC | TODO | TODO | TODO | TODO |

After inserting the measured values, compare each configuration with its Q1
counterpart using both absolute time and percentage difference. The model and
training workload are the same, so any material difference should be discussed
in terms of loader sharding, framework bookkeeping, and run-to-run variability.
Irrespective of timing, Accelerate reduces distributed boilerplate and makes the
accumulation loop easier to read and maintain.

## 3. Profiling

### 3.1 Linear, BatchNorm, and ReLU

The following values must be copied from the ReLU module table generated by
`Profiler.ipynb` on Colab. Each number covers exactly 100 test batches.

| Module family | Self CPU time | CPU memory attributed by profiler |
|---|---:|---:|
| Linear | TODO | TODO |
| BatchNorm | TODO | TODO |
| ReLU | TODO | TODO |

Linear performs the model's matrix multiplications and is expected to dominate
CPU time. BatchNorm additionally computes and applies normalization statistics.
ReLU is an element-wise maximum and should generally require less computation.
The measured table, rather than this expectation, determines the final ordering.

### 3.2 Activation functions

| Activation | Self CPU time | CPU memory attributed by profiler |
|---|---:|---:|
| ReLU | TODO | TODO |
| Sigmoid | TODO | TODO |
| Tanh | TODO | TODO |
| GeLU | TODO | TODO |

Insert the notebook's activation chart below this table. ReLU uses a simple
piecewise operation, while Sigmoid and Tanh evaluate nonlinear transcendental
functions and GeLU performs a more complex smooth transformation. The final
comparison must use the Colab measurements. Profiler memory is
operator-attributed allocation traffic, not whole-process peak resident memory,
so it should not be interpreted as the application's maximum RAM usage.

## Reproducibility

- Cluster nodes used: TODO
- PyTorch version: TODO
- Accelerate version: TODO
- Colab runtime/CPU: TODO
- Random seed: 1404
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001

## AI-tool disclosure

OpenAI Codex was used to compare the supplied CA4 directory with the assignment
handout and a reference implementation, create the initial solution files,
review the distributed-training logic, and correct the Slurm launch scripts.
The generated code was checked with Python compilation, Bash syntax validation,
local smoke tests, and manual source review. Cluster timings and Colab profiler
measurements must come from the student's own executions and are not generated
or copied from the reference repository.

### Prompts used

1. "Can you check the Documents/uni/term7/DMLS/CA folder to see if there is any
   file that is not in Desktop/distributed-ml-systems? If not, add it and make it
   clean."
2. "Don't add the videos."
3. "The main folder is in uni/term7/DMLS and the folder here is a copy of it."
4. "Aren't they in CA4?"
5. "Can you create the files needed to solve CA4?"
6. "If we should unzip those data, do it."
7. "Can you complete the files yourself?"
8. "Check this GitHub repository for a friend who completed these projects and
   determine whether our solution is incomplete or incorrect."
9. "Can you fix these issues and complete it?"

### Links consulted

- Reference implementation:
  <https://github.com/mvajhi/Distributed-Machine-Learning-Systems/tree/master/CA4>
- Hugging Face Accelerate launch documentation:
  <https://huggingface.co/docs/accelerate/main/basic_tutorials/launch>
- PyTorch DistributedDataParallel documentation:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>

Conversation share link: **ADD THE EXPORTED CONVERSATION LINK BEFORE SUBMISSION.**
