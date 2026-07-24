from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


SOURCE_DIRECTORIES = [
    Path("nibi_configs/final_good_trajectory"),
    Path("nibi_configs/final_good_small_ranges"),
    Path("nibi_configs/final_good_under_radar"),
]

OUTPUT_ROOT = Path(
    "nibi_configs/final_feature_importance_combined"
)

CONFIG_LIST = Path(
    "feature_importance_configs.txt"
)


# Every final raw and trajectory feature, including vehicle ID.
COMBINED_FEATURES = [
    "x_pos",
    "y_pos",

    # Original/raw BSM features
    "coreData_elevation",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
    "coreData_accuracy_semiMajor",

    # Engineered trajectory features
    "traj_step_distance_m",
    "traj_distance_error_m",
    "traj_heading_error_deg",
    "traj_position_prediction_error_m",
    "traj_speed_error_mps",
    "traj_accel_mps2",
    "traj_turn_change_deg",

    # Explicitly included identifier
    "coreData_id",
]


REQUIRED_SOURCE_COLUMNS = [
    "metadata_generatedAt",
    "metadata_receivedAt",
    "coreData_id",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_elevation",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
    "coreData_accuracy_semiMajor",
]


def add_unique(values: list[str], additions: list[str]) -> list[str]:
    result = list(values)

    for value in additions:
        if value not in result:
            result.append(value)

    return result


def number_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def determine_region(config: dict, source: Path) -> str:
    name = (
        config.get("pipeline_name", "")
        + " "
        + source.stem
    ).lower()

    for region in [
        "rock_springs",
        "laramie",
        "evanston",
    ]:
        if region in name:
            return region

    filtering = config.get("data", {}).get("filtering", {})
    latitude = float(filtering.get("center_latitude", 0))
    longitude = float(filtering.get("center_longitude", 0))

    known_centres = {
        "rock_springs": (41.538689, -109.319556),
        "laramie": (41.31, -105.6),
        "evanston": (41.27, -110.96),
    }

    for region, (known_lat, known_lon) in known_centres.items():
        if (
            abs(latitude - known_lat) < 0.01
            and abs(longitude - known_lon) < 0.01
        ):
            return region

    raise RuntimeError(
        f"Could not determine region for {source}"
    )


def determine_attack(config: dict) -> str:
    attack_type = config["attack"]["type"]

    if attack_type in {"rand_offset", "override_rand"}:
        return "rpo"

    if attack_type in {
        "const_offset_per_id",
        "const_offset",
    }:
        return "cpo"

    raise RuntimeError(
        f"Unsupported attack type: {attack_type}"
    )


if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

source_configs: list[Path] = []

for root in SOURCE_DIRECTORIES:
    if not root.exists():
        raise RuntimeError(
            f"Missing source directory: {root}"
        )

    for attack in ["rpo", "cpo"]:
        directory = root / attack / "id"

        if not directory.exists():
            raise RuntimeError(
                f"Missing source config directory: {directory}"
            )

        source_configs.extend(
            sorted(directory.glob("*extended_id.json"))
        )


print("===== SOURCE CONFIG DISCOVERY =====")
print(f"Extended-with-ID source configs: {len(source_configs)}")

if len(source_configs) != 84:
    raise RuntimeError(
        "Expected 84 source extended-with-ID configs "
        f"but found {len(source_configs)}."
    )


created: list[Path] = []
seen_conditions: set[tuple] = set()

for source in source_configs:
    with source.open() as handle:
        config = json.load(handle)

    attack = determine_attack(config)
    region = determine_region(config, source)

    minimum = float(
        config["attack"]["offset_distance_min"]
    )

    maximum = float(
        config["attack"]["offset_distance_max"]
    )

    condition_key = (
        attack,
        region,
        minimum,
        maximum,
    )

    if condition_key in seen_conditions:
        raise RuntimeError(
            f"Duplicate experiment condition: {condition_key}"
        )

    seen_conditions.add(condition_key)

    minimum_tag = number_tag(minimum)
    maximum_tag = number_tag(maximum)

    pipeline_name = (
        f"feature_importance_combined_"
        f"{attack}_{region}_"
        f"{minimum_tag}_{maximum_tag}_with_id"
    )

    new_config = copy.deepcopy(config)

    new_config["pipeline_name"] = pipeline_name
    new_config["version"] = (
        "feature-importance-combined-v1"
    )

    # Correct stale template metadata, including the under-radar
    # source configs whose filenames still mention 0.5-1.5 m.
    template = new_config.setdefault(
        "template_parameters",
        {},
    )

    template["feature_set"] = (
        "combined_raw_trajectory_with_id"
    )
    template["with_vehicle_id"] = True
    template["range_min"] = minimum
    template["range_max"] = maximum
    template["feature_importance_run"] = True

    data = new_config.setdefault("data", {})

    data["columns_to_extract"] = add_unique(
        data.get("columns_to_extract", []),
        REQUIRED_SOURCE_COLUMNS,
    )

    ml = new_config.setdefault("ml", {})

    ml["features"] = list(COMBINED_FEATURES)
    ml["trajectory_mode"] = True
    ml["classifiers"] = [
        "RandomForest",
        "DecisionTree",
        "KNeighbors",
    ]

    # Completely isolate this diagnostic run from previous caches.
    new_config["cache"] = {
        "enabled": False,
        "version": (
            "feature-importance-combined-v1"
        ),
    }

    new_config["feature_importance"] = {
        "enabled": True,
        "method": "test_set_permutation_importance",
        "scoring": "attacker_f1",
        "include_vehicle_id": True,
        "feature_count": len(COMBINED_FEATURES),
        "source_config": str(source),
        "results_dir": (
            "results/feature_importance_combined"
        ),
    }

    new_config["output"] = {
        "results_dir": (
            f"results/feature_importance_pipeline/"
            f"{attack}/{pipeline_name}/"
        ),
        "log_dir": (
            f"logs/feature_importance_pipeline/"
            f"{attack}/"
        ),
    }

    output_directory = OUTPUT_ROOT / attack
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{pipeline_name}.json"
    )

    with output_path.open("w") as handle:
        json.dump(
            new_config,
            handle,
            indent=2,
        )

    created.append(output_path)


created = sorted(created)

CONFIG_LIST.write_text(
    "\n".join(str(path) for path in created)
    + "\n"
)


rpo_count = sum(
    "/rpo/" in str(path)
    for path in created
)

cpo_count = sum(
    "/cpo/" in str(path)
    for path in created
)

print()
print("===== CREATED FEATURE-IMPORTANCE CONFIGS =====")
print(f"RPO configs:       {rpo_count}")
print(f"CPO configs:       {cpo_count}")
print(f"Total configs:     {len(created)}")
print(f"Features/config:   {len(COMBINED_FEATURES)}")
print(f"Models/config:     3")
print(
    "Model-level FI analyses: "
    f"{len(created) * 3}"
)

assert rpo_count == 42
assert cpo_count == 42
assert len(created) == 84
assert len(seen_conditions) == 84

print()
print("SUCCESS")
print(f"Created: {CONFIG_LIST}")
