#!/bin/bash

#SBATCH --job-name='SPECFIT'
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=232GB
#SBATCH --time=14-00:00:00
#SBATCH --output=SPECFITjob-%j-stdout.log
#SBATCH --error=SPECFITjob-%j-stderr.log
#SBATCH --partition=Main

module add mpich/3.3a2
#####export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun singularity exec  /data/exp_soft/containers/ASTRO-PY3.simg python specfit.py
