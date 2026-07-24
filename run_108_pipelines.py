#!/usr/bin/env python3
"""
Full 108-Pipeline Executor for ConnectedDrivingPipelineV4
All permutations: 3 features x 3 radii x 6 attacks x 2 (with/without ID) = 108
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

FEATURE_SETS = {
    "basic": ["x_pos", "y_pos", "coreData_elevation"],
    "movement": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw"],
    "extended": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor"],
}

RADII_CONFIGS = {
    "2km": {"radius_meters": 2000},
    "100km": {"radius_meters": 100000},
    "200km": {"radius_meters": 200000},
}

BASE_CONFIG = {
    "version": "2.0.0",
    "created": "2026-02-24",
    "data": {
        "source_file": "April_2021_Wyoming_Data_Fixed.parquet",
        "source_type": "parquet",
        "columns_to_extract": [
            "metadata_receivedAt",
            "coreData_id",
            "coreData_position_lat",
            "coreData_position_long",
            "coreData_elevation",
            "coreData_speed",
            "coreData_heading",
            "coreData_accelset_accelYaw",
            "coreData_accuracy_semiMajor"
        ],
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
    "cache": {"enabled": True, "version": "v3"},
    "dask": {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "12GB", "dashboard_address": ":0"}
}

def generate_pipeline_config(features_name, radius_name, attack_name, with_id):
    config = json.loads(json.dumps(BASE_CONFIG))
    id_suffix = "_withid" if with_id else ""
    attack_suffix = attack_name.replace("_", "")
    config["pipeline_name"] = f"{features_name}_{radius_name}{id_suffix}_{attack_suffix}"
    features = FEATURE_SETS[features_name].copy()
    if with_id:
        features.append("coreData_id")
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
                for with_id in [False, True]:
                    configs.append(generate_pipeline_config(features_name, radius_name, attack_name, with_id))
    return configs

def run_pipeline(config, results_dir):
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
    
    progress = {"total": len(configs), "completed": 0, "failed": 0, "results": [], "started_at": datetime.now().isoformat()}
    progress_file = Path(results_dir) / "progress_108.json"
    
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
        
        if (i + 1) % 10 == 0:
            pct = 100 * (i + 1) / len(configs)
            c = progress["completed"]
            t = progress["total"]
            f = progress["failed"]
            print(f"\nProgress: {c}/{t} completed ({pct:.1f}%), {f} failed")
    
    sep = "=" * 70
    c = progress["completed"]
    t = progress["total"]
    print(f"\n{sep}\nCOMPLETE: {c}/{t} pipelines succeeded\n{sep}")

if __name__ == "__main__":
    main()
