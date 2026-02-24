#!/usr/bin/env python3
"""
Full 162-Pipeline Executor for ConnectedDrivingPipelineV4
All permutations: 9 features × 3 radii × 6 attacks = 162

9 Feature Sets:
1. basic - x_pos, y_pos, coreData_elevation (NO IDs)
2. movement - + speed, heading, accelYaw (NO IDs)
3. extended - + accuracy_semiMajor (NO IDs)
4. basicWithId - basic + coreData_id (X only)
5. movementWithId - movement + coreData_id (X only)
6. extendedWithId - extended + coreData_id (X only)
7. basicWithAll3Ids - basic + coreData_msgCnt (W) + coreData_id (X) + metadata_receivedAt (O)
8. movementWithAll3Ids - movement + W + X + O
9. extendedWithAll3Ids - extended + W + X + O

Column Mapping (Excel positions):
- O (column 15) = metadata_receivedAt
- W (column 23) = coreData_msgCnt
- X (column 24) = coreData_id

Created: 2026-02-24 by Sophie
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

ATTACK_CONFIGS = {
    "rand_offset": {"type": "rand_offset", "offset_distance_min": 100, "offset_distance_max": 200},
    "const_offset": {"type": "const_offset", "direction_angle": 45, "distance_meters": 150},
    "const_offset_per_id": {"type": "const_offset_per_id", "offset_distance_min": 100, "offset_distance_max": 200},
    "swap_rand": {"type": "swap_rand"},
    "override_const": {"type": "override_const", "direction_angle": 45, "distance_meters": 150},
    "override_rand": {"type": "override_rand", "offset_distance_min": 100, "offset_distance_max": 200},
}

# 9 Feature Sets
FEATURE_SETS = {
    # No IDs (3)
    "basic": ["x_pos", "y_pos", "coreData_elevation"],
    "movement": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw"],
    "extended": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor"],
    
    # With X only (coreData_id) (3)
    "basicWithId": ["x_pos", "y_pos", "coreData_elevation", "coreData_id"],
    "movementWithId": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_id"],
    "extendedWithId": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor", "coreData_id"],
    
    # With all 3 IDs: W (coreData_msgCnt), X (coreData_id), O (metadata_receivedAt) (3)
    "basicWithAll3Ids": ["x_pos", "y_pos", "coreData_elevation", "coreData_msgCnt", "coreData_id", "metadata_receivedAt"],
    "movementWithAll3Ids": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_msgCnt", "coreData_id", "metadata_receivedAt"],
    "extendedWithAll3Ids": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor", "coreData_msgCnt", "coreData_id", "metadata_receivedAt"],
}

RADII_CONFIGS = {
    "2km": {"radius_meters": 2000},
    "100km": {"radius_meters": 100000},
    "200km": {"radius_meters": 200000},
}

# All columns we might need to extract from source
ALL_COLUMNS_TO_EXTRACT = [
    "metadata_receivedAt",
    "coreData_id",
    "coreData_msgCnt",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_elevation",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
    "coreData_accuracy_semiMajor"
]

BASE_CONFIG = {
    "version": "2.0.0",
    "created": "2026-02-24",
    "data": {
        "source_file": "April_2021_Wyoming_Data_Fixed.parquet",
        "source_type": "parquet",
        "columns_to_extract": ALL_COLUMNS_TO_EXTRACT,
        "num_subsection_rows": None,  # Use FULL 13.3M dataset (no sampling)
        "filtering": {
            "center_longitude": -109.319556,
            "center_latitude": 41.538689
        },
        "date_range": {"start": "2021-04-01", "end": "2021-04-30"},
        "coordinate_conversion": {"enabled": True, "method": "local_projection", "output_columns": ["x_pos", "y_pos"]}
    },
    "ml": {
        "label": "isAttacker",
        "train_test_split": {"test_size": 0.2, "random_state": 42, "shuffle": True},
        "classifiers": ["RandomForest", "DecisionTree", "KNeighbors"]
    },
    "attack": {"malicious_ratio": 0.3, "seed": 42, "label_column": "isAttacker"},
    "cache": {"enabled": False, "version": "v4"},  # DISABLED to ensure fresh data
    "dask": {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "12GB", "dashboard_address": ":0"}
}

def generate_pipeline_config(features_name, radius_name, attack_name):
    config = json.loads(json.dumps(BASE_CONFIG))
    attack_suffix = attack_name.replace("_", "")
    config["pipeline_name"] = f"{features_name}_{radius_name}_{attack_suffix}"
    
    features = FEATURE_SETS[features_name].copy()
    features.append("isAttacker")
    config["ml"]["features"] = features
    config["data"]["filtering"]["radius_meters"] = RADII_CONFIGS[radius_name]["radius_meters"]
    config["attack"].update(ATTACK_CONFIGS[attack_name])
    return config

def generate_all_configs():
    configs = []
    for features_name in FEATURE_SETS.keys():
        for radius_name in RADII_CONFIGS.keys():
            for attack_name in ATTACK_CONFIGS.keys():
                configs.append(generate_pipeline_config(features_name, radius_name, attack_name))
    return configs

def run_pipeline(config, results_dir):
    # CRITICAL: Clear singleton instances to ensure fresh providers for each pipeline
    # This fixes the bug where GeneratorPathProvider was being reused across runs
    from ClassTypes.SingletonABCMeta import SingletonABCMeta
    SingletonABCMeta._instances.clear()
    
    from MachineLearning.DaskPipelineRunner import DaskPipelineRunner
    pipeline_name = config["pipeline_name"]
    output_dir = Path(results_dir) / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    sep = "=" * 70
    print(f"\n{sep}\nSTARTING: {pipeline_name}\n{sep}")
    start_time = time.time()
    
    try:
        runner = DaskPipelineRunner(config)
        results, metadata = runner.run_with_metadata()
        elapsed = time.time() - start_time
        
        with open(output_dir / f"{pipeline_name}_results.json", "w") as f:
            json.dump({"pipeline_name": pipeline_name, "config": config, "metadata": metadata, 
                      "elapsed_seconds": elapsed, "completed_at": datetime.now().isoformat()}, f, indent=2, default=str)
        
        print(f"COMPLETED: {pipeline_name} in {elapsed:.1f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"FAILED: {pipeline_name} - {str(e)}")
        with open(output_dir / "error.log", "w") as f:
            import traceback
            f.write(f"Error: {str(e)}\n{traceback.format_exc()}")
        return False, elapsed

def main():
    results_dir = "/var/www/static/pipeline-results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    configs = generate_all_configs()
    print(f"Generated {len(configs)} pipeline configurations")
    print(f"\n9 Feature Sets: {list(FEATURE_SETS.keys())}")
    print(f"3 Radii: {list(RADII_CONFIGS.keys())}")
    print(f"6 Attacks: {list(ATTACK_CONFIGS.keys())}")
    print(f"Total: 9 × 3 × 6 = {len(configs)} pipelines\n")
    
    progress = {"total": len(configs), "completed": 0, "failed": 0, "results": [], "started_at": datetime.now().isoformat()}
    progress_file = Path(results_dir) / "progress_162.json"
    
    for i, config in enumerate(configs):
        pname = config["pipeline_name"]
        print(f"\n[{i+1}/{len(configs)}] Running {pname}")
        
        success, elapsed = run_pipeline(config, results_dir)
        
        if success:
            progress["completed"] += 1
        else:
            progress["failed"] += 1
        
        progress["results"].append({"name": pname, "success": success, "elapsed": elapsed})
        
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
        
        if (i + 1) % 18 == 0:  # Every 18 pipelines (1 full feature set round)
            pct = 100 * (i + 1) / len(configs)
            c = progress["completed"]
            t = progress["total"]
            f_count = progress["failed"]
            print(f"\n{'='*50}\nProgress: {c}/{t} completed ({pct:.1f}%), {f_count} failed\n{'='*50}")
    
    sep = "=" * 70
    c = progress["completed"]
    t = progress["total"]
    print(f"\n{sep}\nCOMPLETE: {c}/{t} pipelines succeeded\n{sep}")

if __name__ == "__main__":
    main()
