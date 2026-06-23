#!/bin/bash
# ============================================================================
# Preempt QOS driver script — Perlmutter GPU, 1 GPU (shared)
#
# Usage:
#   sbatch batch/pm_submit_preempt_1GPU.sh uv run main.py fit --config params/pions_odd.yaml
#
# Preempt QOS billing:
#   - 1× charge for first 2 hours
#   - 0.25× charge after 2 hours (GPU)
#   - Minimum 2 hr walltime required
#   - Max 48 hrs, preemptible after 2 hrs
#   - Max 128 nodes
#
# Requeue behavior:
#   - Preemption (SIGTERM) → --requeue auto-requeues
#   - Timeout (walltime)   → payload traps USR1, calls scontrol requeue
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C "gpu"
#SBATCH -q preempt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/pm-slurm-%j-%x.out
#SBATCH -e slurm_logs/pm-slurm-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1

echo "=============================================="
echo " Job ID:     ${SLURM_JOB_ID}"
echo " QOS:        preempt (1× first 2h, 0.25× after)"
echo " Node:       $(hostname)"
echo " Walltime:   ${SLURM_TIMELIMIT}"
echo " Arguments:  $@"
echo "=============================================="

mkdir -p slurm_logs

# Run the payload via srun.  --signal goes to srun's child (the payload).
# The payload traps USR1 for timeout requeue; SIGTERM/preemption is
# handled automatically by --requeue.
srun batch/preempt_payload.sh "$@"

echo "[$(date +%H:%M:%S)] srun exited"
sleep 120  # keep job alive so slurm records PREEMPTED state on preemption
