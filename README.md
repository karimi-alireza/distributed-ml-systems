# Distributed Machine Learning Systems

Coursework for **Distributed Machine Learning Systems** at the University of Tehran, Faculty of Electrical and Computer Engineering (Fall 2025).

The assignments were run on the course's own infrastructure: a **Slurm/MPI CPU cluster**, a multi-GPU server, and a **Spark + HDFS cluster of Raspberry Pis**. They cover the systems side of large-scale ML: parallel decomposition, federated learning, custom CUDA kernels, data-parallel training, and big-data pipelines.

## Assignments

| # | Folder | What I built | Tech |
|---|--------|--------------|------|
| CA1 | [MPI & federated learning](ca1-mpi-federated-learning/) | π estimation parallelised with **block vs. cyclic** work distribution under deliberately unbalanced per-term cost; **FedAvg** logistic regression with one server and three clients on non-IID data (1 round × 10 epochs vs. 10 rounds × 1 epoch), plus a malicious-client experiment; determinant and inverse benchmarks comparing **OpenBLAS and Apple Accelerate** | Python, mpi4py, Slurm, PyTorch, NumPy |
| CA2 | [CUDA & PyTorch DDP](ca2-cuda-and-pytorch-ddp/) | A **fused Conv2d + ReLU CUDA kernel** exposed to PyTorch as an autograd function and benchmarked against cuDNN, cuDNN-off and CPU on 2- and 10-layer CNNs; an STL-10 CNN trained on one GPU and then two GPUs with **DistributedDataParallel**, sweeping batch size (16–128) and **gloo vs. NCCL** backends | CUDA C++, PyTorch, torch.distributed |
| CA3 | [Spark & HDFS](ca3-spark-hdfs/) | Video-game sales analytics with **pure RDD** transformations; a **Word2Vec** model on a Wikipedia corpus (nearest neighbours, king − man + woman ≈ queen); a Titanic pipeline that reads from **HDFS**, runs exploratory aggregations, trains a Spark ML logistic regression, and writes the accuracy back to HDFS | PySpark (RDD, SQL, ML), HDFS |
| CA4 | [Torchrun, Accelerate & profiling](ca4-torchrun-and-pytorch-ddp/) | A CPU-only feed-forward classifier trained across one or two Slurm nodes with **torchrun/DDP** and **Hugging Face Accelerate**; communication-efficient gradient accumulation, four topology experiments, rank-zero checkpointing, and per-module CPU time/memory profiling | PyTorch, Gloo, Slurm, Accelerate |

## Project – Communication-Efficient Distributed SGD

[project-communication-efficient-sgd/](project-communication-efficient-sgd/) compares strategies for reducing the communication bottleneck of data-parallel training:

- **QSGD:** unbiased stochastic gradient quantisation
- **EF-signSGD:** 1-bit sign compression with error feedback
- **Local SGD:** synchronise only every few steps

Each is compared against synchronous SGD on a small CNN and on **ResNet-18 / CIFAR-10**, with a simulated network cost model. Final ResNet-18 results (3 epochs):

| Method | Test accuracy | Simulated time |
|--------|--------------:|---------------:|
| Synchronous SGD | 37.5% | 8,600 s |
| QSGD | 35.0% | 1,091 s |
| **Local SGD** | **39.3%** | **579 s** |
| EF-signSGD | 22.8% | 286 s |

Local SGD matched synchronous SGD's accuracy with about 15× less simulated time. The notebooks, the slides and a full report (Persian) are in the folder.

## Other written work

- [research-talks/](research-talks/): a series of five paper-review talks on distributed deep learning. Topics are hybrid data/model parallelism (DAPPLE, FlexFlow, Alpa), automatic parallelization search (AutoDDL, Aceso), heterogeneous LLM serving (Helix), network-aware device placement (NEST), and prefetching in sharded training.
- [pipeline-parallelism-report/](pipeline-parallelism-report/): a report comparing **1F1B** and **zero-bubble** pipeline-parallel schedules.
- [homework/](homework/): two theory assignments covering batching, floating-point formats and endianness, interconnect topologies, collective-communication libraries (MPI, Gloo, NCCL), systolic arrays, and more.

## Verified locally

- CA1: the serial and both MPI π programs agree (π ≈ 3.1415907), and the cyclic split removes the load imbalance (0.7 s / 3.5 s per process → 2.0 s / 2.0 s). FedAvg and the malicious-client run complete with one server and three workers.
- CA3: the RDD analysis in `VideoGameSales.ipynb` was re-executed with local Spark and reproduces the notebook's results.
- CA4: the data loader, profiler, single-process training, and two-process Gloo/DDP and Accelerate paths were exercised locally; the default model reached 84.4% test accuracy after five epochs. The multi-node Slurm topologies still require the course cluster environment.
- The GPU and cluster parts (CA2, the CA3 HDFS job) need the course hardware; their recorded outputs are in the notebooks and reports.

## Running

```bash
# CA1 – any MPI installation
cd ca1-mpi-federated-learning/1.3 && mpirun -n 4 python pi_parallel_v2.py
cd ../2.2 && mpirun -n 4 python logreg_fedavg.py --num_rounds 10 --num_epochs 1   # needs ../Data/

# CA2 – two GPUs
python ca2-cuda-and-pytorch-ddp/classifier_mp.py --batch_size 32 --backend nccl
```

The CA1 client datasets and the CA3 CSVs were provided by the course and aren't included.
