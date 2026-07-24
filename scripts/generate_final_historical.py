from __future__ import annotations

import copy
import json
from pathlib import Path


GOOD_ROOT = Path("nibi_configs/final_good_trajectory")
OUTPUT_ROOT = Path("nibi_configs/final_bad_historical")

LOCATIONS = [
    "rock_springs",
    "laramie",
    "evanston",
]

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
    "basic": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
    ],

    "movement": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
    ],

    "extended": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
        "coreData_accuracy_semiMajor",
    ],

    "basicWithId": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_id",
    ],

    "movementWithId": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
        "coreData_id",
    ],

    "extendedWithId": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
        "coreData_accuracy_semiMajor",
        "coreData_id",
    ],

    "basicWithAll3Ids": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_msgCnt",
        "coreData_id",
        "metadata_receivedAt",
    ],

    "movementWithAll3Ids": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
        "coreData_msgCnt",
        "coreData_id",
        "metadata_receivedAt",
    ],

    "extendedWithAll3Ids": [
        "x_pos",
        "y_pos",
        "coreData_elevation",
        "coreData_speed",
        "coreData_heading",
        "coreData_accelset_accelYaw",
        "coreData_accuracy_semiMajor",
        "coreData_msgCnt",
        "coreData_id",
        "metadata_receivedAt",
    ],
}

RAW_COLUMNS = [
    "metadata_generatedAt",
    "metadata_receivedAt",
    "coreData_id",
    "coreData_msgCnt",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_elevation",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
    "coreData_accuracy_semiMajor",
]


def get_group(feature_name):
    if feature_name.endswith("WithAll3Ids"):
        return "all3ids"
    if feature_name.endswith("WithId"):
        return "id"
    return "core"


def main():
    grand_total = 0

    for attack in ["rpo", "cpo"]:
        counts = {
            "core": 0,
            "id": 0,
            "all3ids": 0,
        }

        for group in counts:
            directory = OUTPUT_ROOT / attack / group
            directory.mkdir(parents=True, exist_ok=True)

            for old_file in directory.glob("*.json"):
                old_file.unlink()

        for location in LOCATIONS:
            for minimum, maximum in RANGES:

                template_path = (
                    GOOD_ROOT
                    / attack
                    / "core"
                    / (
                        f"final_good_{attack}_"
                        f"{location}_"
                        f"{minimum}_{maximum}_xy.json"
                    )
                )

                if not template_path.exists():
                    raise FileNotFoundError(
                        f"Missing template: {template_path}"
                    )

                with template_path.open("r") as f:
                    template = json.load(f)

                for feature_name, features in FEATURE_SETS.items():
                    cfg = copy.deepcopy(template)

                    pipeline_name = (
                        f"final_historical_{attack}_"
                        f"{location}_"
                        f"{minimum}_{maximum}_"
                        f"{feature_name}"
                    )

                    cfg["pipeline_name"] = pipeline_name
                    cfg["version"] = (
                        "final-historical-signed-xy-2026-07-23"
                    )

                    # Historical/raw model inputs.
                    cfg["ml"]["features"] = list(features)

                    # Keep the same trajectory-eligible population
                    # as the corrected GOOD experiments.
                    cfg["ml"]["trajectory_mode"] = True

                    # Never allow the target into X.
                    if "isAttacker" in cfg["ml"]["features"]:
                        raise RuntimeError(
                            f"Label leakage in {pipeline_name}"
                        )

                    columns = cfg["data"].setdefault(
                        "columns_to_extract",
                        [],
                    )

                    for column in RAW_COLUMNS:
                        if column not in columns:
                            columns.append(column)

                    cfg.setdefault("cache", {})
                    cfg["cache"]["enabled"] = False
                    cfg["cache"]["version"] = (
                        "final-historical-v1"
                    )

                    cfg.setdefault("output", {})

                    cfg["output"]["results_dir"] = (
                        "results/final_bad_historical/"
                        f"{attack}/{pipeline_name}/"
                    )

                    cfg["output"]["log_dir"] = (
                        "logs/final_bad_historical/"
                        f"{attack}/"
                    )

                    params = cfg.setdefault(
                        "template_parameters",
                        {},
                    )

                    params["feature_set"] = feature_name
                    params["historical_raw_features"] = True

                    group = get_group(feature_name)

                    output_path = (
                        OUTPUT_ROOT
                        / attack
                        / group
                        / f"{pipeline_name}.json"
                    )

                    with output_path.open("w") as f:
                        json.dump(
                            cfg,
                            f,
                            indent=2,
                        )

                    counts[group] += 1

        total = sum(counts.values())

        print(f"\n{attack.upper()}:")
        print(f"  core:     {counts['core']}")
        print(f"  id:       {counts['id']}")
        print(f"  all3ids:  {counts['all3ids']}")
        print(f"  Total:    {total}")
        print(f"  Evaluations: {total * 3}")

        assert counts["core"] == 72
        assert counts["id"] == 72
        assert counts["all3ids"] == 72
        assert total == 216

        grand_total += total

    assert grand_total == 432

    print()
    print("=" * 60)
    print("HISTORICAL CONFIG GENERATION PASSED")
    print("432 configs")
    print("1,296 classifier evaluations")
    print("=" * 60)


if __name__ == "__main__":
    main()
