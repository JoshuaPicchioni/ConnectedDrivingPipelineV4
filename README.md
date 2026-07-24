# ConnectedDrivingPipelineV4

A config-driven, Dask-based research pipeline for processing large connected-vehicle Basic Safety Message (BSM) datasets, injecting controlled position-falsification attacks, engineering temporal trajectory-consistency features, and evaluating classical machine-learning detectors.

> **Project lineage**
>
> This repository is a research fork of the ConnectedDrivingPipelineV4 work originally started by **Aaron Collins**. The current fork has been substantially extended by **Joshua Picchioni** for thesis research involving real-world Wyoming Connected Vehicle Pilot data, vehicle-disjoint evaluation, deterministic attack injection, trajectory-consistency features, large experiment sweeps, audit tooling, and feature-importance analysis.

## Project Status

**Status:** Active research software  
**Primary use case:** Misbehaviour detection for connected-vehicle position falsification  
**Current platform:** Python 3.11, Dask, scikit-learn, Apache Arrow/Parquet, Slurm  
**Primary compute environment:** Digital Research Alliance of Canada Nibi cluster  
**Last major research update:** July 2026

This repository should be treated as a reproducible research pipeline rather than a general-purpose production service. Many scripts and configuration directories correspond to specific thesis experiments and are intentionally retained for traceability.

## Research Overview

The pipeline evaluates whether position-falsification attacks can be detected when they are injected into **real connected-vehicle telemetry** rather than fully simulated traffic.

The principal dataset is the April 2021 Wyoming Connected Vehicle Pilot BSM collection:

- approximately **13.3 million source records**
- approximately **69 source columns**
- stored as Parquet for scalable processing
- analyzed in three 100 km regional subsets:
  - Rock Springs
  - Laramie
  - Evanston

The current research focuses on two attack mechanisms:

- **Random Position Offset (RPO):** each malicious BSM receives a newly sampled position offset.
- **Constant Position Offset (CPO):** each malicious vehicle receives one deterministic position-offset vector that remains fixed across its messages.

These attacks behave differently under temporal analysis:

- RPO disrupts message-to-message trajectory consistency.
- CPO translates an entire trajectory while largely preserving its local motion.

That distinction is central to the final experiments.

## Major Capabilities

- Unified JSON-configured `DaskPipelineRunner`
- Dask-based processing of multi-million-row BSM datasets
- Parquet-based intermediate storage and cache isolation
- Signed local Cartesian coordinate projection
- Vehicle-level attacker assignment
- Vehicle-disjoint train/test splitting
- Deterministic RPO and CPO generation
- Temporal trajectory construction by vehicle and timestamp
- Raw BSM and engineered trajectory feature families
- Random Forest, Decision Tree, and K-Nearest Neighbors evaluation
- Accuracy, precision, recall, specificity, confusion matrix, and F1 reporting
- Train/test generalization-gap analysis
- Natural trajectory-residual analysis
- Test-set permutation feature importance
- Slurm array-job generation and execution
- Pipeline and attack audit modes
- Reproducible experiment extraction into consolidated CSV files

## Important Differences from the Original Pipeline

The current fork includes several research-critical changes beyond the earlier Dask migration:

1. **Vehicle-disjoint evaluation**

   Unique `coreData_id` values are partitioned before model training. All BSM records from a vehicle remain entirely in the training or test partition. This prevents the same temporary vehicle identity and adjacent trajectory records from appearing on both sides of the evaluation boundary.

2. **Global vehicle-level attacker assignment**

   Attacker status is assigned at the vehicle level before row-level attack generation. The configured attacker ratio therefore represents compromised vehicles rather than independently selected messages.

3. **Signed regional coordinates**

   Latitude and longitude are projected into signed local planar coordinates:

   - east: positive `x_pos`
   - west: negative `x_pos`
   - north: positive `y_pos`
   - south: negative `y_pos`

4. **Deterministic CPO injection**

   CPO offsets are reproducible and stable per vehicle, independent of Dask partition order.

5. **Trajectory-consistency engineering**

   The pipeline now evaluates whether consecutive positions are consistent with reported speed, heading, elapsed time, acceleration, and turn behaviour.

6. **Cache isolation and validation**

   Cache identities incorporate pipeline versions, selected columns, extracted support columns, and experiment configuration. This prevents stale or incompatible cached datasets from being reused silently.

7. **Runtime auditing**

   Optional audits verify:

   - train/test vehicle separation
   - binary attacker labels
   - required feature columns
   - finite model matrices
   - unchanged benign positions
   - configured attack ranges
   - constant per-vehicle CPO vectors
   - reproducible experiment structure

8. **Under-the-radar attack testing**

   The displacement sweep now extends down to **0.1–0.5 m**, allowing the models to be tested when injected offsets approach the scale of natural short-term trajectory variation.

9. **Held-out permutation feature importance**

   The repository includes a model-agnostic feature-importance workflow using attacker-class F1 on a deterministic sample of the held-out test set.

## Feature Families

### Historical Raw BSM Features

These configurations use unmodified fields already present in each BSM.

| Reporting name | Legacy config name | Features |
|---|---|---|
| Raw Spatial | `basic` | `x_pos`, `y_pos`, `coreData_elevation` |
| Raw Vehicle State | `movement` | Raw Spatial + `coreData_speed`, `coreData_heading`, `coreData_accelset_accelYaw` |
| Raw Vehicle State and Accuracy | `extended` | Raw Vehicle State + `coreData_accuracy_semiMajor` |

Historical experiments also include variants with:

- `coreData_id`
- `coreData_msgCnt`
- `metadata_receivedAt`

These identifier-related fields are analyzed separately because they do not directly represent physical vehicle behaviour.

### Trajectory-Consistency Features

The thesis and current documentation use descriptive reporting names to avoid confusion with the historical raw feature tiers.

| Reporting name | Legacy config name | Features |
|---|---|---|
| Position Baseline | `xy` | `x_pos`, `y_pos` |
| Step Displacement | `basic` | Position Baseline + `traj_step_distance_m` |
| Motion Consistency | `movement` | Step Displacement + `traj_distance_error_m`, `traj_heading_error_deg` |
| Full Kinematic Consistency | `extended` | Motion Consistency + `traj_position_prediction_error_m`, `traj_speed_error_mps`, `traj_accel_mps2`, `traj_turn_change_deg` |

Current engineered columns:

- `traj_step_distance_m`
- `traj_distance_error_m`
- `traj_heading_error_deg`
- `traj_position_prediction_error_m`
- `traj_speed_error_mps`
- `traj_accel_mps2`
- `traj_turn_change_deg`

Support fields such as timestamps, speed, heading, original position, and vehicle identity may be required to calculate trajectory features without automatically being passed into the classifier.

### Identifier Variants

Trajectory configurations are evaluated:

- without `coreData_id`
- with `coreData_id`

The identifier is useful for grouping messages into trajectories, but direct inclusion as a numeric classifier feature can cause overfitting under vehicle-disjoint evaluation.

## Experiment Matrix

The completed thesis result matrix contains **3,312 classifier-level evaluations** before the separate feature-importance sweep.

### Original displacement ranges

- 5–15 m
- 10–30 m
- 30–70 m
- 50–150 m
- 100–200 m
- 150–250 m
- 200–400 m
- 400–600 m

### Small-displacement ranges

- 0.5–1.5 m
- 1–3 m
- 2–5 m
- 3–7 m
- 5–10 m

### Under-the-radar range

- 0.1–0.5 m

### Shared dimensions

- 2 attack types: RPO and CPO
- 3 regions: Rock Springs, Laramie, Evanston
- 3 classifiers: Random Forest, Decision Tree, KNN
- raw and trajectory feature families
- with-ID and no-ID variants where applicable

## Key Research Findings

The repository contains the exact configs and logs used to produce the thesis result tables. The high-level findings are:

- Adding plausible raw BSM fields does not automatically improve attack detection.
- Raw speed, heading, yaw acceleration, and positional accuracy can reduce generalization when they are supplied independently of the attacked position.
- Explicit trajectory-consistency features strongly improve RPO detection.
- CPO remains difficult because a constant translation preserves local trajectory shape.
- Direct numeric inclusion of `coreData_id` substantially increases train/test generalization gaps.
- Small RPO offsets remain detectable by tree models when temporal inconsistencies are exposed.
- KNN becomes substantially weaker in the smallest under-the-radar conditions.
- No single feature representation is sufficient for every position-falsification mechanism.

## Natural Trajectory Residual Analysis

`analyze_natural_position_offness.py` estimates a one-step kinematic position-prediction residual using:

- previous reported position
- previous reported speed
- previous reported heading
- elapsed time
- next reported position

This residual is **not GPS ground-truth error**. It measures the combined real-world variation that a short-term trajectory detector must distinguish from an injected attack.

For the trajectory-valid population used in the analysis:

- observations: approximately 7.80 million
- median residual: approximately 0.25 m
- mean residual: approximately 0.75 m
- 95th percentile: approximately 2.49 m
- 99th percentile: approximately 4.41 m

The 0.1–0.5 m and 0.5–1.5 m experiments therefore stress the detector in ranges that overlap strongly with ordinary short-term variation.

## Feature Importance

The feature-importance workflow combines all final physical and trajectory variables, including `coreData_id`, and evaluates them for each trained classifier.

Current combined feature list:

```text
x_pos
y_pos
coreData_elevation
coreData_speed
coreData_heading
coreData_accelset_accelYaw
coreData_accuracy_semiMajor
traj_step_distance_m
traj_distance_error_m
traj_heading_error_deg
traj_position_prediction_error_m
traj_speed_error_mps
traj_accel_mps2
traj_turn_change_deg
coreData_id
```

Method:

- held-out test-set permutation importance
- attacker-class F1 scorer
- deterministic stratified test sample
- repeated shuffling
- Random Forest, Decision Tree, and KNN
- optional native tree importance retained for comparison

Relevant files:

```text
scripts/generate_feature_importance_configs.py
scripts/run_feature_importance.py
feature_importance_array.slurm
nibi_configs/final_feature_importance_combined/
results/feature_importance_combined/
```

## Repository Layout

The exact contents may evolve, but the main research paths are:

```text
ConnectedDrivingPipelineV4/
├── Generator/
│   ├── Attackers/
│   └── Cleaners/
├── Gatherer/
├── MachineLearning/
│   ├── DaskPipelineRunner.py
│   ├── DaskMClassifierPipeline.py
│   └── MDataClassifier.py
├── Helpers/
├── Logger/
├── ServiceProviders/
├── scripts/
├── nibi_configs/
│   ├── final_good_trajectory/
│   ├── final_good_small_ranges/
│   ├── final_good_under_radar/
│   ├── final_bad_historical/
│   └── final_feature_importance_combined/
├── slurm_feature_importance/
├── results/
├── run_config.py
└── README.md
```

Important research scripts include:

```text
scripts/generate_final_good.py
scripts/generate_final_historical.py
scripts/generate_small_range_good.py
scripts/generate_under_radar.py
scripts/generate_feature_importance_configs.py
scripts/run_feature_importance.py
extract_final_good_results.py
analyze_natural_position_offness.py
```

## Requirements

### Software

- Python 3.11 recommended
- Dask and Distributed
- pandas
- NumPy
- scikit-learn
- PyArrow
- matplotlib
- joblib
- python-dateutil

Install the repository requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Hardware

Requirements depend heavily on the selected region, feature set, classifier, and cache state.

For the full Wyoming experiments, the pipeline has been run using:

- 12 CPU cores per Slurm task
- approximately 56–64 GB RAM per task
- local scratch storage
- Parquet source and cache files

Small local smoke tests can use fewer resources, but the full experiment matrix is intended for a high-memory workstation or HPC cluster.

## Data

The full Wyoming dataset is not included in this repository.

Expected thesis source path on Nibi:

```text
$SCRATCH/wyoming_april_2021/data/April_2021_Wyoming_Data_Fixed.parquet
```

Example source characteristics:

```text
Rows: approximately 13,318,200
Columns: 69
Format: Parquet
```

Users must obtain and prepare an appropriate BSM dataset separately and update config paths accordingly.

## Running on Nibi

Log in:

```bash
ssh <username>@nibi.alliancecan.ca
```

Move to the repository and load the tested environment:

```bash
cd $SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0

source ~/connected-driving-env/bin/activate
```

Run one JSON configuration interactively:

```bash
python -u run_config.py path/to/config.json
```

Enable validation audits:

```bash
export PIPELINE_AUDIT=1
export PIPELINE_ATTACK_AUDIT=1

python -u run_config.py path/to/config.json
```

## Slurm Example

A typical array task follows this pattern:

```bash
#!/bin/bash
#SBATCH --job-name=connected-driving
#SBATCH --account=<allocation>
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=08:00:00
#SBATCH --array=0-47%8
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.out

set -uo pipefail

cd "$SCRATCH/wyoming_april_2021/ConnectedDrivingPipelineV4" || exit 1

module load StdEnv/2023
module load python/3.11
module load arrow/25.0.0
source ~/connected-driving-env/bin/activate

export PIPELINE_AUDIT=1
export PIPELINE_ATTACK_AUDIT=1

CONFIG=$(
    sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs.txt
)

python -u run_config.py "$CONFIG"
```

Submit:

```bash
sbatch experiment_array.slurm
```

Monitor:

```bash
squeue -u "$USER"
sacct -j <job-id> -X --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

## Running Feature Importance

Generate the combined configs:

```bash
python scripts/generate_feature_importance_configs.py
```

Syntax-check the runner:

```bash
python -m py_compile scripts/run_feature_importance.py
```

Submit the Slurm array:

```bash
sbatch feature_importance_array.slurm
```

Expected output directory:

```text
results/feature_importance_combined/
```

Each completed condition writes:

- one CSV containing feature-level importance values
- one JSON containing run metadata and model summaries

## Configuration Notes

Experiment configurations are JSON files consumed by:

```python
from MachineLearning.DaskPipelineRunner import DaskPipelineRunner

runner = DaskPipelineRunner.from_config("path/to/config.json")
results = runner.run()
```

For workflows that need fitted classifiers, exact test matrices, or split metadata:

```python
results, metadata = runner.run_with_metadata()
```

Configuration fields vary between historical and final experiment generations. Current research configs generally include:

- pipeline name and version
- source file
- regional filtering
- source/support columns
- selected classifier features
- attacker ratio
- attack type
- offset range
- random seed
- vehicle-disjoint split behaviour
- cache identity
- output paths
- optional feature-importance metadata

Use the existing generated configs as the authoritative examples for the current schema.

## Reproducibility

The final experiments use deterministic settings wherever practical:

- random seed: 42
- vehicle-level attack assignment
- vehicle-disjoint train/test split
- deterministic per-vehicle CPO vectors
- deterministic RPO configuration
- Decision Tree `random_state=42`
- Random Forest `random_state=42`
- cache versioning
- explicit model-boundary feature selection
- attack and matrix audits
- saved JSON configs
- saved Slurm logs
- consolidated result CSVs

For exact reproduction, preserve:

1. the source dataset version
2. the JSON config
3. the repository commit
4. the Python environment
5. the Slurm resource allocation
6. audit environment variables
7. generated logs and result files

## Validation and Auditing

### Pipeline audit

```bash
export PIPELINE_AUDIT=1
```

Checks model matrices for issues such as:

- missing configured features
- unexpected support columns
- nonnumeric values
- NaN or infinite values
- label/feature inconsistencies

### Attack audit

```bash
export PIPELINE_ATTACK_AUDIT=1
```

Checks properties such as:

- binary labels
- benign positions unchanged
- attacker offsets within the configured interval
- constant CPO displacement per vehicle
- train and test attack generation

### Vehicle-disjoint verification

Final runs report the number of train and test vehicle IDs and verify that their intersection is empty.

## Result Files

Major consolidated outputs include:

```text
final_good_all_results.csv
all_final_experiment_results.csv
natural_offness_summary.csv
natural_offness_thresholds.csv
natural_offness_vehicle_summary.csv
```

The combined result table includes:

- experiment group
- attack type
- region
- displacement range
- feature set
- identifier variant
- classifier
- train metrics
- test metrics
- pipeline/config references
- log references

## Known Limitations

- Attacks are synthetically injected into real telemetry.
- The evaluation uses one connected-vehicle deployment.
- CPO cannot be reliably exposed by local trajectory consistency alone.
- Temporary vehicle identifiers are dataset-specific and nonphysical.
- KNN is sensitive to feature scaling and high-dimensional numeric inputs.
- Feature importance describes how a fitted model uses the evaluated dataset; it is not a universal causal ranking.
- Natural kinematic residual is not GPS ground-truth error.
- Large experiments require substantial RAM, CPU time, and storage.
- Some research directories contain legacy names retained for result reproducibility.

## Development Guidance

Before committing a new experiment:

1. create a unique pipeline name
2. assign a new cache/version identifier
3. explicitly define model features
4. keep support columns separate from model inputs
5. use a deterministic random seed
6. run both audit modes
7. validate config counts
8. run one smoke-test task
9. inspect the complete log
10. launch the full Slurm array
11. verify expected result counts
12. archive configs, logs, scripts, and extracted results

Basic syntax checks:

```bash
python -m py_compile MachineLearning/DaskPipelineRunner.py
python -m py_compile scripts/run_feature_importance.py
```

Repository status:

```bash
git status
git diff --stat
```

## Contributing

Research contributions and reproducibility fixes are welcome.

Recommended workflow:

```bash
git checkout -b feature/descriptive-name
git add .
git commit -m "Describe the research or pipeline change"
git push -u origin HEAD
```

Please include:

- the motivation for the change
- affected config directories
- whether cached results are invalidated
- validation commands
- smoke-test results
- expected experiment counts

## Attribution

### Aaron Collins

Aaron Collins started the original ConnectedDrivingPipelineV4 work and its earlier pandas/Dask pipeline direction. This repository exists as a fork and continuation of that foundation.

### Joshua Picchioni

Joshua Picchioni developed and validated the current thesis-oriented research fork, including major work on:

- Wyoming April 2021 dataset processing
- signed local-coordinate handling
- vehicle-level attacker assignment
- vehicle-disjoint evaluation
- deterministic RPO/CPO generation
- trajectory-consistency features
- cache isolation and configuration hashing
- model-boundary feature selection
- runtime audits
- large Slurm experiment sweeps
- under-the-radar displacement testing
- natural trajectory-residual analysis
- consolidated result extraction
- permutation feature-importance tooling
- thesis result interpretation and documentation

## Acknowledgments

This work uses data derived from the Wyoming Connected Vehicle Pilot and open-source software including Dask, pandas, NumPy, scikit-learn, and Apache Arrow.

Development and debugging have also been assisted by automated coding and analysis tools. All final research decisions, experiment definitions, validation steps, and thesis interpretations remain the responsibility of the project authors.

## License

No license should be assumed unless a license file is present in the repository. Add or update `LICENSE` before distributing or accepting external contributions under specific terms.
