#!/bin/bash
#SBATCH --job-name=all3ids-smoke
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=all3ids_smoke_%j.out

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

mkdir -p all3ids_smoke_logs

RPO_CONFIG="nibi_configs/phase0_5/phase0_5_rock_springs_rpo_movement_with_all3ids_50_75_rf.json"
CPO_CONFIG="nibi_configs/phase0_5/phase0_5_rock_springs_cpo_movement_with_all3ids_50_75_rf.json"

srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=56G \
    python -u run_config.py "$RPO_CONFIG" \
    > all3ids_smoke_logs/rpo.out 2>&1 &

RPO_PID=$!

srun --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --mem=56G \
    python -u run_config.py "$CPO_CONFIG" \
    > all3ids_smoke_logs/cpo.out 2>&1 &

CPO_PID=$!

wait "$RPO_PID"
RPO_STATUS=$?

wait "$CPO_PID"
CPO_STATUS=$?

echo "RPO exit status: $RPO_STATUS"
echo "CPO exit status: $CPO_STATUS"

if [ "$RPO_STATUS" -ne 0 ] || [ "$CPO_STATUS" -ne 0 ]; then
    exit 1
fi

echo "ALL3IDS SMOKE TEST PASSED"
