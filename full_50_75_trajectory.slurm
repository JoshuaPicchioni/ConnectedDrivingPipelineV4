#!/bin/bash
#SBATCH --job-name=full-50-75-traj
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=full_50_75_trajectory_%j.out

set -uo pipefail

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export RF_N_JOBS=12
export VEHICLE_EQUAL_WEIGHT=0
export PIPELINE_AUDIT=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOG_DIR="full_50_75_trajectory_logs/job_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

pids=()
names=()

for cfg in nibi_configs/full_50_75_trajectory/*.json; do
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
success=0

for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "SUCCESS: ${names[$i]}"
        success=$((success + 1))
    else
        echo "FAILED: ${names[$i]}"
        failed=$((failed + 1))
    fi
done

echo "================================"
echo "Successful configs: $success"
echo "Failed configs:     $failed"
echo "Expected configs:   24"
echo "================================"

exit "$failed"
