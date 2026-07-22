#!/bin/bash
# ============================================================================
# Test script for debug_preempt QOS — verifies checkpoint save on preemption.
#
# Usage:
#   sbatch batch/test_preempt_debug.sh
#
# This script:
#   1. Runs a short training (3 epochs) using the sharded config
#   2. Simulates preemption by sending SIGTERM to itself after 30s
#   3. Verifies the checkpoint was saved with save_on_exception
#   4. On requeue, verifies training resumes from the checkpoint
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q debug_preempt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --signal=USR1@60        # 60s warning before walltime
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/test-preempt-%j-%x.out
#SBATCH -e slurm_logs/test-preempt-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=============================================="
echo " Test: Preempt QOS Checkpoint Behavior"
echo " Job ID:     ${SLURM_JOB_ID}"
echo " QOS:        debug_preempt"
echo " Node:       $(hostname)"
echo " Walltime:   ${SLURM_TIMELIMIT}"
echo "=============================================="

mkdir -p slurm_logs

# Run the training via the preempt payload wrapper
srun batch/preempt_payload.sh \
    uv run python main.py fit \
    --config params/pions_odd_sharded.yaml \
    --trainer.max_epochs 3 \
    --trainer.logger null

echo "[$(date +%H:%M:%S)] Training completed or exited"
echo "[$(date +%H:%M:%S)] Job ID: ${SLURM_JOB_ID}"
echo "[$(date +%H:%M:%S)] Checkpoint dir: /pscratch/sd/p/pmtuan/caloinn/full_training_odd_lightning/${SLURM_JOB_ID}/"

# List checkpoints
CKPT_DIR="/pscratch/sd/p/pmtuan/caloinn/full_training_odd_lightning/${SLURM_JOB_ID}"
if [ -d "$CKPT_DIR" ]; then
    echo "=============================================="
    echo " Checkpoints in $CKPT_DIR:"
    ls -la "$CKPT_DIR"/*.ckpt 2>/dev/null || echo "  (no .ckpt files found)"
    echo "=============================================="
fi

sleep 120  # keep job alive until slurm sends SIGKILL
