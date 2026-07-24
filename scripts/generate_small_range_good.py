from __future__ import annotations

import copy
import json
from pathlib import Path


SOURCE_ROOT = Path(
    "nibi_configs/final_good_trajectory"
)

OUTPUT_ROOT = Path(
    "nibi_configs/final_good_small_ranges"
)

NEW_RANGES = [
    (0.5, 1.5),
    (1.0, 3.0),
    (2.0, 5.0),
    (3.0, 7.0),
    (5.0, 10.0),
]


def number_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def main():
    grand_total = 0

    for attack in ["rpo", "cpo"]:

        attack_total = 0

        for group in ["core", "id"]:

            source_dir = (
                SOURCE_ROOT
                / attack
                / group
            )

            output_dir = (
                OUTPUT_ROOT
                / attack
                / group
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Remove only previously generated
            # SMALL-RANGE configs.
            # Existing final_good_trajectory configs
            # and results are untouched.
            for old in output_dir.glob(
                "*.json"
            ):
                old.unlink()

            # Use all existing 5-15 GOOD configs
            # as templates.
            templates = sorted(
                source_dir.glob(
                    "*_5_15_*.json"
                )
            )

            print(
                f"{attack.upper()} "
                f"{group}: "
                f"{len(templates)} templates"
            )

            # Expected:
            # 3 locations × 4 feature sets
            # = 12 templates per group.
            if len(templates) != 12:
                raise RuntimeError(
                    f"Expected 12 templates in "
                    f"{source_dir}, "
                    f"found {len(templates)}"
                )

            for template_path in templates:

                with template_path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:

                    template = json.load(
                        handle
                    )

                old_name = template[
                    "pipeline_name"
                ]

                # Replace only the existing
                # _5_15_ section.
                if "_5_15_" not in old_name:
                    raise RuntimeError(
                        f"Unexpected template name: "
                        f"{old_name}"
                    )

                for (
                    minimum,
                    maximum,
                ) in NEW_RANGES:

                    cfg = copy.deepcopy(
                        template
                    )

                    min_label = number_label(
                        minimum
                    )

                    max_label = number_label(
                        maximum
                    )

                    range_label = (
                        f"{min_label}_"
                        f"{max_label}"
                    )

                    new_name = old_name.replace(
                        "_5_15_",
                        f"_{range_label}_",
                        1,
                    )

                    # Add a prefix so these can never
                    # collide with the original GOOD runs.
                    new_name = new_name.replace(
                        "final_good_",
                        "final_good_small_",
                        1,
                    )

                    cfg[
                        "pipeline_name"
                    ] = new_name

                    cfg[
                        "version"
                    ] = (
                        "final-good-small-ranges-"
                        "2026-07-23"
                    )

                    # Change ONLY attack magnitude.
                    cfg[
                        "attack"
                    ][
                        "offset_distance_min"
                    ] = minimum

                    cfg[
                        "attack"
                    ][
                        "offset_distance_max"
                    ] = maximum

                    # Keep caches isolated/disabled.
                    cfg.setdefault(
                        "cache",
                        {},
                    )

                    cfg[
                        "cache"
                    ][
                        "enabled"
                    ] = False

                    cfg[
                        "cache"
                    ][
                        "version"
                    ] = (
                        "final-good-small-ranges-v1"
                    )

                    # Separate output location.
                    cfg.setdefault(
                        "output",
                        {},
                    )

                    cfg[
                        "output"
                    ][
                        "results_dir"
                    ] = (
                        "results/"
                        "final_good_small_ranges/"
                        f"{attack}/"
                        f"{new_name}/"
                    )

                    cfg[
                        "output"
                    ][
                        "log_dir"
                    ] = (
                        "logs/"
                        "final_good_small_ranges/"
                        f"{attack}/"
                    )

                    params = cfg.setdefault(
                        "template_parameters",
                        {},
                    )

                    params[
                        "small_range_extension"
                    ] = True

                    params[
                        "range_min"
                    ] = minimum

                    params[
                        "range_max"
                    ] = maximum

                    output_path = (
                        output_dir
                        / f"{new_name}.json"
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

                    attack_total += 1

        print()
        print(
            f"{attack.upper()} configs: "
            f"{attack_total}"
        )

        print(
            f"{attack.upper()} evaluations: "
            f"{attack_total * 3}"
        )

        if attack_total != 120:
            raise RuntimeError(
                f"Expected 120 "
                f"{attack.upper()} configs, "
                f"found {attack_total}"
            )

        grand_total += attack_total

    if grand_total != 240:
        raise RuntimeError(
            f"Expected 240 total configs, "
            f"found {grand_total}"
        )

    print()
    print("=" * 60)
    print(
        "SMALL-RANGE CONFIG "
        "GENERATION PASSED"
    )
    print(
        "240 configs"
    )
    print(
        "720 classifier evaluations"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
