#!/bin/bash
#SBATCH --job-name=good-traj-smoke
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=final_good_smoke_%j.out

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4"

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export RF_N_JOBS=12
export VEHICLE_EQUAL_WEIGHT=0
export PIPELINE_AUDIT=1
export PIPELINE_ATTACK_AUDIT=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p final_good_smoke_logs

RPO_CONFIG="nibi_configs/final_good_trajectory/rpo/core/final_good_rpo_rock_springs_5_15_extended.json"
CPO_CONFIG="nibi_configs/final_good_trajectory/cpo/core/final_good_cpo_rock_springs_5_15_extended.json"

srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=56G \
    python -u run_config.py "$RPO_CONFIG" \
    > final_good_smoke_logs/rpo.out 2>&1 &

RPO_PID=$!

srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=56G \
    python -u run_config.py "$CPO_CONFIG" \
    > final_good_smoke_logs/cpo.out 2>&1 &

CPO_PID=$!

wait "$RPO_PID"
RPO_STATUS=$?

wait "$CPO_PID"
CPO_STATUS=$?

echo "RPO exit status: $RPO_STATUS"
echo "CPO exit status: $CPO_STATUS"

if [ "$RPO_STATUS" -ne 0 ] || \
   [ "$CPO_STATUS" -ne 0 ]; then
    exit 1
fi

echo "FINAL GOOD TRAJECTORY SMOKE PASSED"
