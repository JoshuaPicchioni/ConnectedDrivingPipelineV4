#!/bin/bash
#SBATCH --job-name=wy-baseline
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=nibi_baseline_%j.out

set -euo pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4"

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

python -u run_config.py \
    nibi_configs/nibi_baseline_basic_100km_rpo_50_75_rf.json
