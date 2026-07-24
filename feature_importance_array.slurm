#!/bin/bash
#SBATCH --job-name=feature-importance
#SBATCH --account=def-arunita_cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --array=0-83%6
#SBATCH --output=slurm_feature_importance/fi_%A_%a.out
#SBATCH --error=slurm_feature_importance/fi_%A_%a.out

set -uo pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4" \
    || exit 1

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export MPLBACKEND=Agg
export PYTHONHASHSEED=42

# Avoid nested BLAS/thread oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Allow RF and KNN to use the CPUs assigned by Slurm.
export RF_N_JOBS="${SLURM_CPUS_PER_TASK}"

# Final feature-importance settings.
export FI_MAX_TEST_ROWS=5000
export FI_N_REPEATS=10
export FI_RANDOM_SEED=42

# Retain the existing pipeline integrity checks.
export PIPELINE_AUDIT=1
export PIPELINE_ATTACK_AUDIT=1

CONFIG=$(
    sed -n \
    "$((SLURM_ARRAY_TASK_ID + 1))p" \
    feature_importance_configs.txt
)

if [ -z "${CONFIG}" ]; then
    echo "ERROR: No config for array task ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config does not exist: ${CONFIG}"
    exit 1
fi

echo "============================================================"
echo "FEATURE-IMPORTANCE ARRAY TASK"
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Array task:      ${SLURM_ARRAY_TASK_ID}"
echo "Config:          ${CONFIG}"
echo "Started:         $(date)"
echo "Maximum rows:    ${FI_MAX_TEST_ROWS}"
echo "Repetitions:     ${FI_N_REPEATS}"
echo "============================================================"

python -u scripts/run_feature_importance.py \
    "${CONFIG}"

STATUS=$?

echo
echo "============================================================"
echo "Finished:        $(date)"
echo "Exit status:     ${STATUS}"
echo "Config:          ${CONFIG}"
echo "============================================================"

exit "${STATUS}"
