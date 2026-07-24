#!/bin/bash
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --array=0-215%8
#SBATCH --output=slurm_final_bad/%x_%A_%a.out

set -euo pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4"

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export RF_N_JOBS=12
export VEHICLE_EQUAL_WEIGHT=0
export PIPELINE_AUDIT=0
export PIPELINE_ATTACK_AUDIT=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ -z "${CONFIG_LIST:-}" ]; then
    echo "ERROR: CONFIG_LIST missing"
    exit 1
fi

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
CONFIG=$(sed -n "${LINE_NUMBER}p" "$CONFIG_LIST")

if [ -z "$CONFIG" ]; then
    echo "ERROR: Missing config"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "============================================================"
echo "FINAL HISTORICAL FEATURE RUN"
echo "Job:     $SLURM_JOB_ID"
echo "Task:    $SLURM_ARRAY_TASK_ID"
echo "Config:  $CONFIG"
echo "Started: $(date)"
echo "============================================================"

python -u run_config.py "$CONFIG"

STATUS=$?

echo
echo "============================================================"
echo "Finished:    $(date)"
echo "Exit status: $STATUS"
echo "Config:      $CONFIG"
echo "============================================================"

exit "$STATUS"
