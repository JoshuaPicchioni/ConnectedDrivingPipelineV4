#!/usr/bin/env python3
"""Rerun movementWithAll3Ids_2km_randoffset pipeline"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

ATTACK_CONFIG = {"type": "rand_offset", "offset_distance_min": 100, "offset_distance_max": 200}

FEATURES = ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", 
            "coreData_accelset_accelYaw", "coreData_msgCnt", "coreData_id", "metadata_receivedAt", "isAttacker"]

ALL_COLUMNS = [
    "metadata_receivedAt", "coreData_id", "coreData_msgCnt", "coreData_position_lat",
    "coreData_position_long", "coreData_elevation", "coreData_speed", "coreData_heading",
    "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor"
]

config = {
    "version": "2.0.0",
    "created": "2026-02-26",
    "pipeline_name": "movementWithAll3Ids_2km_randoffset",
    "data": {
        "source_file": "April_2021_Wyoming_Data_Fixed.parquet",
        "source_type": "parquet",
        "columns_to_extract": ALL_COLUMNS,
        "num_subsection_rows": None,
        "filtering": {
            "center_longitude": -109.319556,
            "center_latitude": 41.538689,
            "radius_meters": 2000
        },
        "date_range": {"start": "2021-04-01", "end": "2021-04-30"},
        "coordinate_conversion": {"enabled": True, "method": "local_projection", "output_columns": ["x_pos", "y_pos"]}
    },
    "ml": {
        "label": "isAttacker",
        "features": FEATURES,
        "train_test_split": {"test_size": 0.2, "random_state": 42, "shuffle": True},
        "classifiers": ["RandomForest", "DecisionTree", "KNeighbors"]
    },
    "attack": {"malicious_ratio": 0.3, "seed": 42, "label_column": "isAttacker", **ATTACK_CONFIG},
    "cache": {"enabled": False, "version": "v4"},
    "dask": {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "12GB", "dashboard_address": ":0"}
}

if __name__ == "__main__":
    from ClassTypes.SingletonABCMeta import SingletonABCMeta
    SingletonABCMeta._instances.clear()
    
    from MachineLearning.DaskPipelineRunner import DaskPipelineRunner
    
    results_dir = "/var/www/static/pipeline-results"
    pipeline_name = config["pipeline_name"]
    output_dir = Path(results_dir) / pipeline_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting {pipeline_name} rerun...")
    start_time = time.time()
    
    try:
        runner = DaskPipelineRunner(config)
        results, metadata = runner.run_with_metadata()
        elapsed = time.time() - start_time
        
        with open(output_dir / f"{pipeline_name}_results.json", "w") as f:
            json.dump({"pipeline_name": pipeline_name, "config": config, "metadata": metadata,
                      "elapsed_seconds": elapsed, "completed_at": datetime.now().isoformat()}, f, indent=2, default=str)
        
        print(f"SUCCESS: {pipeline_name} completed in {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        print(f"FAILED: {pipeline_name} - {str(e)}")
        traceback.print_exc()
        with open(output_dir / "error.log", "w") as f:
            f.write(f"Error: {str(e)}\n{traceback.format_exc()}")
