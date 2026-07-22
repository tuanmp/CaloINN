#!/usr/bin/env bash
# Gather raw SLURM cluster data for Claude to format into a readable summary.
# Outputs structured sections separated by markers.
set -euo pipefail

echo "=== CLUSTER_NAME ==="
scontrol show config 2>/dev/null | awk -F'= ' '/ClusterName/{print $2}' || echo "unknown"

echo "=== PARTITION_OVERVIEW ==="
sinfo -o "%P %G %c %m %z %l %a %D %N" --noheader 2>/dev/null

echo "=== PARTITION_DETAILS ==="
scontrol show partition 2>/dev/null

echo "=== NODE_TYPES ==="
sinfo -N -o "%N %c %m %z %G %T" --noheader 2>/dev/null | sort -u -k1,1

echo "=== QOS_LIMITS ==="
sacctmgr show qos format=Name%12,MaxWall%14,MaxTRESPerUser%20,MaxJobsPerUser%12,MaxSubmitJobsPerUser%14 2>/dev/null

echo "=== NODE_AVAILABILITY ==="
sinfo -o "%P %F" --noheader 2>/dev/null

echo "=== END ==="
