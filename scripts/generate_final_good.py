from __future__ import annotations

import copy
import json
from pathlib import Path


SOURCE_DIR = Path("nibi_configs/full_50_75_trajectory")
OUTPUT_ROOT = Path("nibi_configs/final_good_trajectory")

LOCATIONS = {
    "rock_springs": {
        "longitude": -109.319556,
        "latitude": 41.538689,
    },
    "laramie": {
        "longitude": -105.6000,
        "latitude": 41.3100,
    },
    "evanston": {
        "longitude": -110.9600,
        "latitude": 41.2700,
    },
}

ATTACKS = {
    "rpo": "rand_offset",
    "cpo": "const_offset_per_id",
}

# Same eight attack ranges used in the previous mega sweep.
RANGES = [
    (5, 15),
    (10, 30),
    (30, 70),
    (50, 150),
    (100, 200),
    (150, 250),
    (200, 400),
    (400, 600),
]

FEATURE_SETS = {
    "xy": [
        "x_pos",
        "y_pos",
    ],

    "basic": [
        "x_pos",
        "y_pos",
        "traj_step_distance_m",
    ],

    "movement": [
        "x_pos",
        "y_pos",
        "traj_step_distance_m",
        "traj_distance_error_m",
        "traj_heading_error_deg",
    ],

    "extended": [
        "x_pos",
        "y_pos",
        "traj_step_distance_m",
        "traj_distance_error_m",
        "traj_heading_error_deg",
        "traj_position_prediction_error_m",
        "traj_speed_error_mps",
        "traj_accel_mps2",
        "traj_turn_change_deg",
    ],
}

CLASSIFIERS = [
    "RandomForest",
    "DecisionTree",
    "KNeighbors",
]

# These columns are needed internally to construct trajectory features.
# They are NOT automatically supplied to sklearn.
SUPPORT_COLUMNS = [
    "metadata_generatedAt",
    "coreData_id",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
]


def ensure_support_columns(cfg: dict) -> None:
    data = cfg.setdefault("data", {})

    columns = data.setdefault(
        "columns_to_extract",
        [],
    )

    for column in SUPPORT_COLUMNS:
        if column not in columns:
            columns.append(column)


def build_config(
    template: dict,
    attack_label: str,
    expected_attack_type: str,
    location: str,
    range_min: int,
    range_max: int,
    feature_name: str,
    feature_columns: list[str],
    include_id: bool,
) -> dict:

    cfg = copy.deepcopy(template)

    variant = (
        f"{feature_name}_id"
        if include_id
        else feature_name
    )

    pipeline_name = (
        f"final_good_{attack_label}_"
        f"{location}_"
        f"{range_min}_{range_max}_"
        f"{variant}"
    )

    cfg["pipeline_name"] = pipeline_name
    cfg["version"] = (
        "final-signed-xy-2026-07-23"
    )

    # --------------------------------------------------------
    # Location validation
    # --------------------------------------------------------

    filtering = (
        cfg
        .setdefault("data", {})
        .setdefault("filtering", {})
    )

    expected_location = LOCATIONS[location]

    actual_lon = float(
        filtering["center_longitude"]
    )

    actual_lat = float(
        filtering["center_latitude"]
    )

    if abs(
        actual_lon
        - expected_location["longitude"]
    ) > 1e-6:
        raise RuntimeError(
            f"{location}: unexpected longitude "
            f"{actual_lon}"
        )

    if abs(
        actual_lat
        - expected_location["latitude"]
    ) > 1e-6:
        raise RuntimeError(
            f"{location}: unexpected latitude "
            f"{actual_lat}"
        )

    # Always use 100 km regional filtering.
    filtering["radius_meters"] = 100000

    # --------------------------------------------------------
    # Attack
    # --------------------------------------------------------

    attack = cfg.setdefault(
        "attack",
        {},
    )

    actual_attack_type = attack.get(
        "type"
    )

    if actual_attack_type != expected_attack_type:
        raise RuntimeError(
            f"{pipeline_name}: expected attack "
            f"{expected_attack_type}, got "
            f"{actual_attack_type}"
        )

    attack[
        "offset_distance_min"
    ] = range_min

    attack[
        "offset_distance_max"
    ] = range_max

    attack[
        "malicious_ratio"
    ] = 0.3

    attack[
        "seed"
    ] = 42

    # --------------------------------------------------------
    # ML features
    # --------------------------------------------------------

    ml = cfg.setdefault(
        "ml",
        {},
    )

    features = list(
        feature_columns
    )

    if include_id:
        if "coreData_id" not in features:
            features.append(
                "coreData_id"
            )

    # Critical safety assertion:
    # the target label must NEVER enter X.
    if "isAttacker" in features:
        raise RuntimeError(
            f"{pipeline_name}: "
            f"isAttacker leaked into ML features"
        )

    ml["features"] = features

    ml[
        "classifiers"
    ] = CLASSIFIERS

    ml[
        "trajectory_mode"
    ] = True

    ml[
        "label"
    ] = "isAttacker"

    ml[
        "train_test_split"
    ] = {
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": True,
        "type": "vehicle_disjoint",
        "train_ratio": 0.8,
    }

    ensure_support_columns(
        cfg
    )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    cache = cfg.setdefault(
        "cache",
        {},
    )

    # The runner has its own corrected config/code-hash cache
    # identity. Disable config-level legacy cache reuse anyway.
    cache["enabled"] = False
    cache["version"] = (
        "final-signed-xy-v4"
    )

    cache[
        "clean_dataset"
    ] = (
        f"cache/final_good_trajectory/"
        f"{attack_label}/"
        f"{pipeline_name}/"
        f"clean.parquet"
    )

    cache[
        "attack_dataset"
    ] = (
        f"cache/final_good_trajectory/"
        f"{attack_label}/"
        f"{pipeline_name}/"
        f"attack.parquet"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = cfg.setdefault(
        "output",
        {},
    )

    output[
        "results_dir"
    ] = (
        f"results/final_good_trajectory/"
        f"{attack_label}/"
        f"{pipeline_name}/"
    )

    output[
        "log_dir"
    ] = (
        f"logs/final_good_trajectory/"
        f"{attack_label}/"
    )

    # --------------------------------------------------------
    # Template metadata
    # --------------------------------------------------------

    template_parameters = (
        cfg.setdefault(
            "template_parameters",
            {},
        )
    )

    template_parameters[
        "attack_type"
    ] = expected_attack_type

    template_parameters[
        "feature_set"
    ] = feature_name

    template_parameters[
        "with_vehicle_id"
    ] = include_id

    template_parameters[
        "malicious_ratio"
    ] = 0.3

    return cfg


def main() -> None:

    total_created = 0

    for attack_label, attack_type in ATTACKS.items():

        attack_root = (
            OUTPUT_ROOT
            / attack_label
        )

        core_dir = (
            attack_root
            / "core"
        )

        id_dir = (
            attack_root
            / "id"
        )

        core_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        id_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Remove stale generated configs.
        for directory in (
            core_dir,
            id_dir,
        ):
            for old_path in directory.glob(
                "*.json"
            ):
                old_path.unlink()

        core_count = 0
        id_count = 0

        for location in LOCATIONS:

            template_path = (
                SOURCE_DIR
                / (
                    f"full_50_75_"
                    f"{location}_"
                    f"{attack_label}_"
                    f"xy.json"
                )
            )

            if not template_path.exists():
                raise FileNotFoundError(
                    f"Missing template: "
                    f"{template_path}"
                )

            with template_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                template = json.load(
                    handle
                )

            for range_min, range_max in RANGES:

                range_label = (
                    f"{range_min}_"
                    f"{range_max}"
                )

                for (
                    feature_name,
                    feature_columns,
                ) in FEATURE_SETS.items():

                    # ------------------------------
                    # WITHOUT ID
                    # ------------------------------

                    cfg = build_config(
                        template=template,
                        attack_label=attack_label,
                        expected_attack_type=attack_type,
                        location=location,
                        range_min=range_min,
                        range_max=range_max,
                        feature_name=feature_name,
                        feature_columns=feature_columns,
                        include_id=False,
                    )

                    path = (
                        core_dir
                        / (
                            f"final_good_"
                            f"{attack_label}_"
                            f"{location}_"
                            f"{range_label}_"
                            f"{feature_name}.json"
                        )
                    )

                    with path.open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(
                            cfg,
                            handle,
                            indent=2,
                        )

                    core_count += 1

                    # ------------------------------
                    # WITH coreData_id
                    # ------------------------------

                    cfg_id = build_config(
                        template=template,
                        attack_label=attack_label,
                        expected_attack_type=attack_type,
                        location=location,
                        range_min=range_min,
                        range_max=range_max,
                        feature_name=feature_name,
                        feature_columns=feature_columns,
                        include_id=True,
                    )

                    path_id = (
                        id_dir
                        / (
                            f"final_good_"
                            f"{attack_label}_"
                            f"{location}_"
                            f"{range_label}_"
                            f"{feature_name}_id.json"
                        )
                    )

                    with path_id.open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(
                            cfg_id,
                            handle,
                            indent=2,
                        )

                    id_count += 1

        print()
        print(
            f"{attack_label.upper()}:"
        )
        print(
            f"  Core configs: {core_count}"
        )
        print(
            f"  ID configs:   {id_count}"
        )
        print(
            f"  Total:        "
            f"{core_count + id_count}"
        )
        print(
            f"  Evaluations:  "
            f"{(core_count + id_count) * 3}"
        )

        assert core_count == 96
        assert id_count == 96

        total_created += (
            core_count
            + id_count
        )

    assert total_created == 384

    print()
    print("=" * 60)
    print(
        "FINAL GOOD CONFIG GENERATION PASSED"
    )
    print(
        "384 configs total"
    )
    print(
        "1,152 classifier evaluations total"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
