#!/bin/bash
#SBATCH --job-name=matrix-audit
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=matrix_audit_%j.out

set -uo pipefail

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export RF_N_JOBS=12
export PIPELINE_AUDIT=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOG_DIR="matrix_audit_logs/job_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

for cfg in nibi_configs/matrix_audit/*.json; do
    name=$(basename "$cfg" .json)

    srun \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=12 \
        --mem=56G \
        python -u run_config.py "$cfg" \
        > "$LOG_DIR/${name}.out" 2>&1 &
done

wait
