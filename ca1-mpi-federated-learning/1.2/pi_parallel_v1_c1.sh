#!/bin/bash
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=2           # Request 2 tasks (cores) per node 
#SBATCH --output=pi_parallel_v1_c1.out 
#SBATCH --job-name=karimi-pi-v1-c1  

module load GCC Python OpenMPI
time srun --mpi=pmix_v5 python pi_parallel_v1.py