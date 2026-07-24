#!/bin/bash
#SBATCH --job-name=phase0_5
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=phase0_5_%j.out

set -uo pipefail

echo "============================================================"
echo "PHASE 0.5 VEHICLE-DISJOINT VALIDATION"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

# Each experiment receives exactly 12 RF workers.
export RF_N_JOBS=12

# Avoid nested BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG_DIR="nibi_configs/phase0_5"
LOG_DIR="phase0_5_logs/job_${SLURM_JOB_ID}"

mkdir -p "$LOG_DIR"

pids=()
names=()

echo "Launching 30 Phase 0.5 experiments."
echo "Slurm resources allow up to 8 experiment steps concurrently."
echo ""

for cfg in "$CONFIG_DIR"/*.json; do
    name=$(basename "$cfg" .json)

    echo "QUEUEING: $name at $(date)"

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

echo ""
echo "============================================================"
echo "PHASE 0.5 COMPLETE"
echo "============================================================"
echo "Successful experiments: $success"
echo "Failed experiments: $failed"
echo "End: $(date)"

if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
