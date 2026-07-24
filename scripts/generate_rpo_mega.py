from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SOURCE_DIR = Path("nibi_configs/full_50_75_trajectory")
OUTPUT_ROOT = Path("nibi_configs/rpo_mega_trajectory")
CORE_DIR = OUTPUT_ROOT / "core"
ID_DIR = OUTPUT_ROOT / "id"

LOCATIONS = [
    "rock_springs",
    "laramie",
    "evanston",
]

# Full original RPO sweep.
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

SUPPORT_COLUMNS = [
    "metadata_generatedAt",
    "coreData_id",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_speed",
    "coreData_heading",
    "coreData_accelset_accelYaw",
]


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_50_75(a: Any, b: Any) -> bool:
    return (
        is_number(a)
        and is_number(b)
        and float(a) == 50.0
        and float(b) == 75.0
    )


def find_rand_attack_blocks(
    obj: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], dict]]:
    """
    Find dictionaries that directly identify the random positional-offset attack.
    """
    found = []

    if isinstance(obj, dict):
        immediate_strings = [
            str(v).lower()
            for v in obj.values()
            if isinstance(v, str)
        ]

        marker_found = any(
            (
                "rand_offset" in value
                or "positional_offset_rand" in value
                or "random_offset" in value
            )
            for value in immediate_strings
        )

        if marker_found:
            found.append((path, obj))

        for key, value in obj.items():
            found.extend(
                find_rand_attack_blocks(
                    value,
                    path + (str(key),),
                )
            )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(
                find_rand_attack_blocks(
                    value,
                    path + (str(index),),
                )
            )

    return found


def patch_50_75_inside_attack_block(
    block: Any,
    new_min: int,
    new_max: int,
) -> int:
    """
    Replace the existing 50-75 range inside the identified RPO attack block.

    Supports:
      min/max key pairs
      [50, 75] style range arrays
      fallback unique numeric 50 and 75 values
    """
    changes = 0

    def walk_named(obj: Any) -> None:
        nonlocal changes

        if isinstance(obj, dict):
            keys = list(obj.keys())

            min_keys = []
            max_keys = []

            for key in keys:
                normalized = str(key).lower()

                relevant = any(
                    token in normalized
                    for token in (
                        "offset",
                        "distance",
                        "meter",
                        "metre",
                    )
                )

                if relevant and "min" in normalized:
                    min_keys.append(key)

                if relevant and "max" in normalized:
                    max_keys.append(key)

            for min_key in min_keys:
                for max_key in max_keys:
                    if is_50_75(
                        obj[min_key],
                        obj[max_key],
                    ):
                        obj[min_key] = new_min
                        obj[max_key] = new_max
                        changes += 1

            for key, value in obj.items():
                normalized = str(key).lower()

                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and is_50_75(value[0], value[1])
                    and any(
                        token in normalized
                        for token in (
                            "range",
                            "offset",
                            "distance",
                        )
                    )
                ):
                    value[0] = new_min
                    value[1] = new_max
                    changes += 1
                else:
                    walk_named(value)

        elif isinstance(obj, list):
            for value in obj:
                walk_named(value)

    walk_named(block)

    if changes > 0:
        return changes

    # Fallback:
    # Locate unique numeric 50 and 75 values inside the RPO attack subtree.
    hits_50 = []
    hits_75 = []

    def collect(
        obj: Any,
        parent: Any = None,
        key: Any = None,
    ) -> None:
        if is_number(obj):
            if float(obj) == 50.0:
                hits_50.append((parent, key))
            elif float(obj) == 75.0:
                hits_75.append((parent, key))
            return

        if isinstance(obj, dict):
            for child_key, value in obj.items():
                collect(value, obj, child_key)

        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                collect(value, obj, index)

    collect(block)

    if len(hits_50) == 1 and len(hits_75) == 1:
        parent_50, key_50 = hits_50[0]
        parent_75, key_75 = hits_75[0]

        parent_50[key_50] = new_min
        parent_75[key_75] = new_max

        return 1

    raise RuntimeError(
        "Could not safely locate the existing 50-75 RPO range "
        f"inside attack block.\n"
        f"50 hits: {len(hits_50)}, 75 hits: {len(hits_75)}\n"
        f"Block:\n{json.dumps(block, indent=2)}"
    )


def patch_rpo_range(
    cfg: dict,
    new_min: int,
    new_max: int,
) -> str:
    blocks = find_rand_attack_blocks(cfg)

    if not blocks:
        raise RuntimeError(
            "Could not locate a rand_offset attack block."
        )

    # Try deepest/most-specific blocks first.
    blocks.sort(
        key=lambda item: len(item[0]),
        reverse=True,
    )

    errors = []

    for path, block in blocks:
        candidate = copy.deepcopy(block)

        try:
            patch_50_75_inside_attack_block(
                candidate,
                new_min,
                new_max,
            )
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        # Patch the real block only after candidate succeeded.
        patch_50_75_inside_attack_block(
            block,
            new_min,
            new_max,
        )

        return ".".join(path) if path else "<root>"

    raise RuntimeError(
        "Found random attack markers, but none contained a "
        "safe 50-75 range to replace.\n\n"
        + "\n\n".join(errors)
    )


def replace_string_recursive(
    obj: Any,
    old: str,
    new: str,
) -> Any:
    if isinstance(obj, dict):
        return {
            key: replace_string_recursive(
                value,
                old,
                new,
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            replace_string_recursive(
                value,
                old,
                new,
            )
            for value in obj
        ]

    if isinstance(obj, str):
        return obj.replace(old, new)

    return obj


def disable_boolean_caches(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).lower()

            if (
                isinstance(value, bool)
                and normalized
                in {
                    "use_cache",
                    "cache_enabled",
                    "enable_cache",
                }
            ):
                obj[key] = False
            else:
                disable_boolean_caches(value)

    elif isinstance(obj, list):
        for value in obj:
            disable_boolean_caches(value)


def ensure_support_columns(cfg: dict) -> None:
    data = cfg.setdefault("data", {})

    for key in ("columns_to_extract", "columns"):
        columns = data.get(key)

        if not isinstance(columns, list):
            continue

        for column in SUPPORT_COLUMNS:
            if column not in columns:
                columns.append(column)


def build_config(
    template: dict,
    location: str,
    range_min: int,
    range_max: int,
    feature_name: str,
    features: list[str],
    include_id: bool,
) -> tuple[dict, str]:
    cfg = copy.deepcopy(template)

    suffix = (
        f"{feature_name}_id"
        if include_id
        else feature_name
    )

    new_pipeline_name = (
        f"rpo_mega_{location}_"
        f"{range_min}_{range_max}_"
        f"{suffix}"
    )

    old_pipeline_name = cfg.get(
        "pipeline_name",
        "",
    )

    if old_pipeline_name:
        cfg = replace_string_recursive(
            cfg,
            old_pipeline_name,
            new_pipeline_name,
        )

    cfg["pipeline_name"] = new_pipeline_name

    ml = cfg.setdefault("ml", {})

    model_features = list(features)

    if include_id:
        if "coreData_id" not in model_features:
            model_features.append("coreData_id")

    ml["features"] = model_features
    ml["classifiers"] = CLASSIFIERS
    ml["trajectory_mode"] = True

    ensure_support_columns(cfg)
    disable_boolean_caches(cfg)

    attack_path = patch_rpo_range(
        cfg,
        range_min,
        range_max,
    )

    return cfg, attack_path


def main() -> None:
    CORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ID_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale generated JSONs before rebuilding.
    for directory in (CORE_DIR, ID_DIR):
        for path in directory.glob("*.json"):
            path.unlink()

    created_core = 0
    created_id = 0
    attack_paths = set()

    for location in LOCATIONS:
        template_path = (
            SOURCE_DIR
            / f"full_50_75_{location}_rpo_xy.json"
        )

        if not template_path.exists():
            raise FileNotFoundError(
                f"Missing validated template: "
                f"{template_path}"
            )

        with template_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            template = json.load(handle)

        for range_min, range_max in RANGES:
            range_label = (
                f"{range_min}_{range_max}"
            )

            for feature_name, features in FEATURE_SETS.items():
                # Priority 1: normal trajectory feature set.
                cfg, attack_path = build_config(
                    template=template,
                    location=location,
                    range_min=range_min,
                    range_max=range_max,
                    feature_name=feature_name,
                    features=features,
                    include_id=False,
                )

                attack_paths.add(attack_path)

                output_path = (
                    CORE_DIR
                    / (
                        f"rpo_mega_{location}_"
                        f"{range_label}_"
                        f"{feature_name}.json"
                    )
                )

                with output_path.open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        cfg,
                        handle,
                        indent=2,
                    )

                created_core += 1

                # Priority 2: exact same feature set + coreData_id.
                cfg_id, attack_path_id = build_config(
                    template=template,
                    location=location,
                    range_min=range_min,
                    range_max=range_max,
                    feature_name=feature_name,
                    features=features,
                    include_id=True,
                )

                attack_paths.add(attack_path_id)

                output_path_id = (
                    ID_DIR
                    / (
                        f"rpo_mega_{location}_"
                        f"{range_label}_"
                        f"{feature_name}_id.json"
                    )
                )

                with output_path_id.open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        cfg_id,
                        handle,
                        indent=2,
                    )

                created_id += 1

    print()
    print("RPO MEGA CONFIG GENERATION COMPLETE")
    print("-----------------------------------")
    print(f"Core configs: {created_core}")
    print(f"ID configs:   {created_id}")
    print(
        f"Total configs: "
        f"{created_core + created_id}"
    )
    print(
        f"Total classifier evaluations: "
        f"{(created_core + created_id) * 3}"
    )
    print(
        "Attack block path(s): "
        + ", ".join(sorted(attack_paths))
    )

    assert created_core == 96
    assert created_id == 96

    print()
    print("VALIDATION PASSED: expected 192 configs.")


if __name__ == "__main__":
    main()
