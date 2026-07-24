#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

# Monkey-patch GeneratorPathProvider to track creation
from ServiceProviders.GeneratorPathProvider import GeneratorPathProvider

original_init = GeneratorPathProvider.__init__
creation_count = [0]

def patched_init(self, contexts=None, model=None):
    creation_count[0] += 1
    print(f"[DEBUG] GeneratorPathProvider.__init__ called #{creation_count[0]}, model={model}, id={id(self)}")
    import traceback
    traceback.print_stack(limit=5)
    original_init(self, contexts, model)

GeneratorPathProvider.__init__ = patched_init

import json
from MachineLearning.DaskPipelineRunner import DaskPipelineRunner

FEATURE_SETS = {
    "basic": ["x_pos", "y_pos", "coreData_elevation"],
}

RADII_CONFIGS = {
    "2km": {"radius_meters": 2000},
    "100km": {"radius_meters": 100000},
}

ATTACK_CONFIGS = {
    "rand_offset": {"type": "rand_offset", "offset_distance_min": 100, "offset_distance_max": 200},
}

ALL_COLUMNS_TO_EXTRACT = [
    "metadata_receivedAt", "coreData_id", "coreData_msgCnt",
    "coreData_position_lat", "coreData_position_long", "coreData_elevation",
    "coreData_speed", "coreData_heading", "coreData_accelset_accelYaw", "coreData_accuracy_semiMajor"
]

BASE_CONFIG = {
    "version": "2.0.0",
    "created": "2026-02-24",
    "data": {
        "source_file": "April_2021_Wyoming_Data_Fixed.parquet",
        "source_type": "parquet",
        "columns_to_extract": ALL_COLUMNS_TO_EXTRACT,
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
    "cache": {"enabled": False, "version": "v4"},
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

print("=== Creating first runner (2km) ===")
config1 = generate_pipeline_config("basic", "2km", "rand_offset")
runner1 = DaskPipelineRunner(config1)

print("\n=== SHOULD CREATE A NEW PROVIDER HERE ===")
print("=== Creating second runner (100km) ===")
config2 = generate_pipeline_config("basic", "100km", "rand_offset")

# Manually call _setup_providers to see what happens
print("\n[MANUAL] Creating GeneratorPathProvider directly:")
test_provider = GeneratorPathProvider(
    model="TEST-MODEL",
    contexts={"test": lambda m: f"path/{m}"}
)
print(f"test_provider created with id={id(test_provider)}, model={test_provider.getModelName()}")

runner2 = DaskPipelineRunner(config2)
print(f"\nrunner2 provider id: {id(runner2._generatorPathProvider)}")
print(f"runner2 provider model: {runner2._generatorPathProvider.getModelName()}")
