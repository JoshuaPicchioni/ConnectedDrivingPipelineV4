#!/usr/bin/env python3
"""Generate pipeline configs for new center points (Laramie and Evanston)."""
import json
import os

CENTERS = {
    "Laramie": {"lat": 41.31, "lon": -105.60},
    "Evanston": {"lat": 41.27, "lon": -110.96},
}

FEATURE_SETS = {
    "basic": {
        "columns": ["metadata_receivedAt", "coreData_position_lat", "coreData_position_long", "coreData_elevation"],
        "features": ["x_pos", "y_pos", "coreData_elevation"],
    },
    "movement": {
        "columns": ["metadata_receivedAt", "coreData_position_lat", "coreData_position_long", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelSet_accelYaw"],
        "features": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelSet_accelYaw"],
    },
    "extended": {
        "columns": ["metadata_receivedAt", "coreData_position_lat", "coreData_position_long", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelSet_accelYaw", "coreData_accuracy_semiMajor"],
        "features": ["x_pos", "y_pos", "coreData_elevation", "coreData_speed", "coreData_heading", "coreData_accelSet_accelYaw", "coreData_accuracy_semiMajor"],
    },
}

ATTACK_TYPES = {
    "constoffset": {
        "type": "const_offset",
        "offset_distance_min": 100,
        "offset_distance_max": 200,
        "offset_direction_min": 0,
        "offset_direction_max": 360,
    },
    "constoffsetperid": {
        "type": "const_offset_per_id",
        "offset_distance_min": 100,
        "offset_distance_max": 200,
        "offset_direction_min": 0,
        "offset_direction_max": 360,
    },
    "randoffset": {
        "type": "rand_offset",
        "offset_distance_min": 100,
        "offset_distance_max": 200,
        "offset_direction_min": 0,
        "offset_direction_max": 360,
    },
    "swaprand": {
        "type": "swap_rand",
    },
}

RADII = {
    "2km": 2000,
    "100km": 100000,
    "200km": 200000,
}

OUTPUT_DIR = "production_configs_v2/new_centers"


def make_config(center_name, center_coords, feature_set_name, feature_set, radius_name, radius_meters, attack_name, attack_params):
    pipeline_name = f"{feature_set_name}_{radius_name}_{attack_name}_center{center_name}"
    
    attack_section = {
        "malicious_ratio": 0.3,
        "label_column": "isAttacker",
        "seed": 42,
    }
    attack_section.update(attack_params)
    
    config = {
        "pipeline_name": pipeline_name,
        "version": "2.0.0",
        "created": "2026-03-26",
        "template_generated": True,
        "template_parameters": {
            "spatial_radius": radius_name,
            "feature_set": feature_set_name.upper(),
            "attack_type": attack_params["type"],
            "with_vehicle_id": False,
            "malicious_ratio": 0.3,
            "center_point": center_name,
            "center_lat": center_coords["lat"],
            "center_lon": center_coords["lon"],
        },
        "data": {
            "source_file": "April_2021_Wyoming_Data_Fixed.parquet",
            "source_type": "parquet",
            "columns_to_extract": feature_set["columns"],
            "num_subsection_rows": None,
            "filtering": {
                "center_longitude": center_coords["lon"],
                "center_latitude": center_coords["lat"],
                "radius_meters": radius_meters,
            },
            "date_range": {
                "start": "2021-04-01",
                "end": "2021-04-30",
            },
            "coordinate_conversion": {
                "enabled": True,
                "method": "local_projection",
                "output_columns": ["x_pos", "y_pos"],
            },
        },
        "attack": attack_section,
        "ml": {
            "features": feature_set["features"],
            "label": "isAttacker",
            "train_test_split": {
                "test_size": 0.2,
                "random_state": 42,
                "shuffle": True,
                "type": "random",
                "train_ratio": 0.8,
            },
            "classifiers": ["RandomForest", "DecisionTree", "KNeighbors"],
        },
        "cache": {
            "enabled": True,
            "version": "v3",
            "clean_dataset": f"cache/matrix/{pipeline_name}/clean.parquet",
            "attack_dataset": f"cache/matrix/{pipeline_name}/attack.parquet",
        },
        "output": {
            "results_dir": f"results/matrix/{pipeline_name}/",
            "log_dir": "logs/",
        },
        "dask": {
            "n_workers": 4,
            "threads_per_worker": 2,
            "memory_limit": "12GB",
            "dashboard_address": ":0",
        },
    }
    return pipeline_name, config


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    configs_generated = []
    
    for center_name, center_coords in CENTERS.items():
        # Standard attacks at all radii
        for attack_name in ["constoffset", "constoffsetperid", "randoffset"]:
            for radius_name, radius_meters in RADII.items():
                for fs_name, fs in FEATURE_SETS.items():
                    pname, config = make_config(
                        center_name, center_coords, fs_name, fs,
                        radius_name, radius_meters, attack_name, ATTACK_TYPES[attack_name]
                    )
                    filepath = os.path.join(OUTPUT_DIR, f"{pname}_pipeline_config.json")
                    with open(filepath, "w") as f:
                        json.dump(config, f, indent=2)
                    configs_generated.append(pname)
        
        # Swaprand: 2km ONLY
        for fs_name, fs in FEATURE_SETS.items():
            pname, config = make_config(
                center_name, center_coords, fs_name, fs,
                "2km", 2000, "swaprand", ATTACK_TYPES["swaprand"]
            )
            filepath = os.path.join(OUTPUT_DIR, f"{pname}_pipeline_config.json")
            with open(filepath, "w") as f:
                json.dump(config, f, indent=2)
            configs_generated.append(pname)
    
    print(f"Generated {len(configs_generated)} configs in {OUTPUT_DIR}/")
    for c in configs_generated:
        print(f"  {c}")


if __name__ == "__main__":
    main()
