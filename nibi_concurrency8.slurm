#!/bin/bash
#SBATCH --job-name=wy-conc8
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=00:30:00
#SBATCH --output=nibi_concurrency8_%j.out

set -euo pipefail

echo "=== 8-Way Concurrent Nibi Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

# Each individual experiment gets 12 CPUs for Random Forest.
# Do NOT let it inherit the full 96-CPU Slurm allocation.
export RF_N_JOBS=12

# Prevent nested numerical libraries from spawning additional threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="nibi_configs/concurrency8"
LOG_DIR="concurrency_logs/job_${SLURM_JOB_ID}"

mkdir -p "$LOG_DIR"

pids=()
names=()

for cfg in "$CONFIG_DIR"/*.json; do
    name=$(basename "$cfg" .json)

    echo "Launching $name at $(date)"

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
        failed=1
    fi
done

echo "End: $(date)"
echo "=== Concurrent benchmark complete ==="

exit "$failed"
