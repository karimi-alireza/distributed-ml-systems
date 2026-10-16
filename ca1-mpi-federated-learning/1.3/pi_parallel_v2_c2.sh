#!/bin/bash
#SBATCH --nodes=2                 
#SBATCH --ntasks-per-node=2           # Request 2 tasks (cores) per node 
#SBATCH --output=pi_parallel_v2_c2.out 
#SBATCH --job-name=karimi-pi-v2-c2  

module load GCC Python OpenMPI
time srun --mpi=pmix_v5 python pi_parallel_v2.py