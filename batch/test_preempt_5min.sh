#!/bin/bash
# ============================================================================
# Test script: 15-min walltime preempt — verifies checkpoint save on SIGTERM.
#
# Usage:
#   sbatch batch/test_preempt_5min.sh
#
# Key: walltime=15min, signal at 120s (2min warning). Queue+setup ~5min,
# training runs ~8min, then signal fires at T+13:00 → SIGUSR1 → preempt_payload
# sends SIGTERM → Lightning saves checkpoint via save_on_exception.
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q debug_preempt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --signal=USR1@120        # 120s warning (2 min) → SIGUSR1 → preempt_payload sends SIGTERM
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/test-preempt-5min-%j-%x.out
#SBATCH -e slurm_logs/test-preempt-5min-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=============================================="
echo " Test: 15-min Preempt QOS Checkpoint Behavior"
echo " Job ID:     ${SLURM_JOB_ID}"
echo " QOS:        debug_preempt"
echo " Node:       $(hostname)"
echo " Walltime:   ${SLURM_TIMELIMIT}"
echo " Signal:     USR1@120"
echo "=============================================="

mkdir -p slurm_logs

# Use a dedicated temp dir to avoid conflict with real training output
CKPT_DIR="/pscratch/sd/p/pmtuan/caloinn/preempt_test_${SLURM_JOB_ID}"
echo "[$(date +%H:%M:%S)] Checkpoint dir: ${CKPT_DIR}"

# Run the training via the preempt payload wrapper
srun batch/preempt_payload.sh \
    uv run python main.py fit \
    --config params/pions_odd_sharded.yaml \
    --trainer.stage_dir /pscratch/sd/p/pmtuan/caloinn \
    --trainer.run_name "preempt_test_${SLURM_JOB_ID}" \
    --trainer.max_epochs 3 \
    --trainer.logger null

RC=$?
echo "[$(date +%H:%M:%S)] Training exited with rc=${RC}"

echo "=============================================="
echo " Job ID:     ${SLURM_JOB_ID}"
echo " Checkpoint dir: ${CKPT_DIR}"
if [ -d "$CKPT_DIR" ]; then
    echo " Directory listing:"
    ls -laR "$CKPT_DIR/"
    echo "---"
    echo " .ckpt files:"
    find "$CKPT_DIR" -name "*.ckpt" -ls 2>/dev/null || echo "  (none)"
else
    echo " WARNING: Checkpoint dir NOT FOUND"
fi
echo "=============================================="

sleep 120  # keep job alive until SLURM sends SIGKILL
