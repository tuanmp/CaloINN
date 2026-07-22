#!/bin/bash
# ============================================================================
# Test: Lightning native HPC fault tolerance — srun directly, no wrapper.
#
# Usage:
#   sbatch batch/test_preempt_native.sh
#
# Flow:
#   1. SLURM sends USR1 120s before walltime (via --signal=USR1@120)
#   2. Lightning's _slurm_sigusr_handler_fn catches USR1
#   3. Saves hpc_ckpt_N.ckpt to default_root_dir/
#   4. Calls scontrol requeue $SLURM_JOB_ID
#   5. Requeued job auto-detects hpc_ckpt_N.ckpt and resumes
#
# Key: running via srun (not a shell wrapper) lets Lightning receive
# SLURM signals directly and register its own handlers.
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q debug_preempt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --signal=USR1@120        # 120s warning → Lightning catches USR1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/test-native-%j-%x.out
#SBATCH -e slurm_logs/test-native-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=============================================="
echo " Test: Lightning Native HPC Fault Tolerance"
echo " Job ID:     ${SLURM_JOB_ID}"
echo " QOS:        debug_preempt"
echo " Node:       $(hostname)"
echo " Walltime:   ${SLURM_TIMELIMIT}"
echo " Signal:     USR1@120 (→ Lightning native handling)"
echo "=============================================="

mkdir -p slurm_logs

CKPT_DIR="/pscratch/sd/p/pmtuan/caloinn/preempt_native_${SLURM_JOB_ID}"
echo "[$(date +%H:%M:%S)] Checkpoint dir: ${CKPT_DIR}"

# Run directly via srun — no preempt_payload shell wrapper.
# Lightning's _SignalConnector receives SLURM signals natively.
srun uv run python main.py fit \
    --config params/pions_odd_sharded.yaml \
    --trainer.stage_dir /pscratch/sd/p/pmtuan/caloinn \
    --trainer.run_name "preempt_native_${SLURM_JOB_ID}" \
    --trainer.max_epochs 3 \
    --trainer.logger null

RC=$?
echo "[$(date +%H:%M:%S)] Training exited with rc=${RC}"

echo "=============================================="
echo " Job ID:     ${SLURM_JOB_ID}"
echo " Checkpoint dir: ${CKPT_DIR}"
if [ -d "$CKPT_DIR" ]; then
    echo " All .ckpt files:"
    find "$CKPT_DIR" -name "*.ckpt" -ls 2>/dev/null || echo "  (none)"
    echo "---"
    echo " hpc_ckpt files:"
    find "$CKPT_DIR" -name "hpc_ckpt_*.ckpt" -ls 2>/dev/null || echo "  (none)"
else
    echo " WARNING: Checkpoint dir NOT FOUND"
fi
echo "=============================================="

sleep 120  # keep job alive until SLURM sends SIGKILL
