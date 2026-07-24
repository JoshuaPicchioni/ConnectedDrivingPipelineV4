#!/bin/bash
#SBATCH --account=def-arunita_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --job-name=offness
#SBATCH --output=natural_offness_%j.out

set -euo pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4"

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

export OMP_NUM_THREADS=8

python -u analyze_natural_position_offness.py
