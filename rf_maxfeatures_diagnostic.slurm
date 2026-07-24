#!/bin/bash
#SBATCH --job-name=rf-maxfeat-diag
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=rf_maxfeatures_diagnostic_%j.out

set -uo pipefail

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export RF_N_JOBS=12
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="nibi_configs/rf_maxfeatures_diagnostic"
LOG_DIR="rf_maxfeatures_diagnostic_logs/job_${SLURM_JOB_ID}"

mkdir -p "$LOG_DIR"

pids=()
names=()

for cfg in "$CONFIG_DIR"/*.json; do
    name=$(basename "$cfg" .json)
    echo "STARTING: $name"

    srun \
        --exclusive \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=12 \
        --mem=56G \
        python -u run_config.py "$cfg" \
        > "$LOG_DIR/${name}.out" 2>&1 &

    pids+=($!)
    names+=("$name")
done

failed=0

for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "SUCCESS: ${names[$i]}"
    else
        echo "FAILED: ${names[$i]}"
        failed=$((failed + 1))
    fi
done

echo "Failed: $failed"
exit "$failed"
