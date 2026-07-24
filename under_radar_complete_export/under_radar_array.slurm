#!/bin/bash
#SBATCH --job-name=under-radar
#SBATCH --account=def-arunita_cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --array=0-47%8
#SBATCH --output=slurm_under_radar/under-radar_%A_%a.out
#SBATCH --error=slurm_under_radar/under-radar_%A_%a.out

set -o pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4" || exit 1

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export PIPELINE_AUDIT=1
export PIPELINE_ATTACK_AUDIT=1

CONFIG=$(
    sed -n \
    "$((SLURM_ARRAY_TASK_ID + 1))p" \
    under_radar_all_configs.txt
)

if [ -z "$CONFIG" ]; then
    echo "ERROR: No config found for task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "============================================================"
echo "UNDER-THE-RADAR EXPERIMENT"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Config:     ${CONFIG}"
echo "Started:    $(date)"
echo "============================================================"

python -u run_config.py "$CONFIG"

STATUS=$?

echo
echo "============================================================"
echo "Finished:    $(date)"
echo "Exit status: ${STATUS}"
echo "Config:      ${CONFIG}"
echo "============================================================"

exit ${STATUS}
