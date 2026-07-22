#!/bin/bash
# ============================================================================
# Preempt payload — runs inside srun, traps signals for graceful requeue.
#
# Receives the user command as arguments from the driver.
# Backgrounds the command so we can trap signals in this shell.
#
# SIGTERM → preemption (--requeue handles auto-requeue)
# USR1    → timeout warning (manually requeue via scontrol)
# ============================================================================

preempt_handler() {
    echo "[$(date +%H:%M:%S)] Received SIGTERM (preemption)"
    echo "[$(date +%H:%M:%S)] Forwarding to child PID ${1}..."
    kill -TERM "${1}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] --requeue will auto-requeue"
    scontrol requeue "${SLURM_JOB_ID}" 
    # catch error from scontrol requeue (e.g., if job is already requeued) and continue
    echo "[$(date +%H:%M:%S)] requeue rc=$?"
}

timeout_handler() {
    echo "[$(date +%H:%M:%S)] Received USR1 (timeout warning)"
    echo "[$(date +%H:%M:%S)] Forwarding to child PID ${1}..."
    kill -TERM "${1}" 2>/dev/null
    echo "[$(date +%H:%M:%S)] Calling scontrol requeue ${SLURM_JOB_ID}..."
    scontrol requeue "${SLURM_JOB_ID}"
    # catch error from scontrol requeue (e.g., if job is already requeued) and continue
    echo "[$(date +%H:%M:%S)] requeue rc=$?"
}

echo "[$(date +%H:%M:%S)] Payload starting on $(hostname)"
echo "[$(date +%H:%M:%S)] Command: $@"

# Run the user command in the background so this shell can trap signals.
# (Matches the NERSC preempt example pattern.)
"$@" &
child_pid=$!

trap "preempt_handler '$child_pid'" SIGTERM
trap "timeout_handler '$child_pid'" USR1

echo "[$(date +%H:%M:%S)] Child PID: ${child_pid}, waiting..."
wait "${child_pid}"

echo "[$(date +%H:%M:%S)] Command exited"
sleep 120  # keep the job step alive until slurm sends SIGKILL
