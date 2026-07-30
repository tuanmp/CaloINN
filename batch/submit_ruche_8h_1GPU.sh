#!/bin/bash
# Submits a CaloINN training job to Ruche GPU partition (8h, 1× V100).
#
# Usage:
#   sbatch batch/submit_ruche_8h_1GPU.sh
#   sbatch batch/submit_ruche_8h_1GPU.sh --config params/lemurs_fcceeallegro.yaml

#SBATCH --job-name=caloinn_train
#SBATCH --output=slurm_logs/ruche-%j-%x.out
#SBATCH --error=slurm_logs/ruche-%j-%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@240
#SBATCH --partition=gpu
#SBATCH --tmp=20G

export HDF5_USE_FILE_LOCKING=FALSE
export SLURM_CPU_BIND="cores"

mkdir -p slurm_logs

# cd ${SLURM_SUBMIT_DIR}

ARG=$@

echo "Starting job: $ARG"
srun $ARG

wait