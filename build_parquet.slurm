#!/bin/bash
#SBATCH --job-name=wy-parquet
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=build_parquet_%j.out

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate

python -u scripts/build_nibi_fixed_parquet.py
