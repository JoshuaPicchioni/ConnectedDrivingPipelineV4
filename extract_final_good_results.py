from pathlib import Path
import json
import re
import pandas as pd


JOBS = {
    "rpo": {
        "job_id": "18292702",
        "pattern": "good-rpo_18292702_*.out",
    },
    "cpo": {
        "job_id": "18292714",
        "pattern": "good-cpo_18292714_*.out",
    },
}

LOG_DIR = Path("slurm_final_good")

LOCATIONS = [
    "rock_springs",
    "laramie",
    "evanston",
]

train_f1_re = re.compile(
    r"TRAIN SET RESULTS:.*?"
    r"F1 Score:\s*([0-9.eE+-]+)",
    re.DOTALL,
)

test_f1_re = re.compile(
    r"TEST SET RESULTS:.*?"
    r"F1 Score:\s*([0-9.eE+-]+)",
    re.DOTALL,
)

config_re = re.compile(
    r"Config:\s*(\S+\.json)"
)

rows = []
problems = []


for attack, info in JOBS.items():

    logs = sorted(
        LOG_DIR.glob(
            info["pattern"]
        )
    )

    print(
        f"{attack.upper()}: "
        f"found {len(logs)} logs"
    )

    for log_path in logs:

        text = log_path.read_text(
            errors="replace"
        )

        # Only accept successfully finished production tasks.
        if "Exit status:     0" not in text:
            problems.append(
                (
                    str(log_path),
                    "missing successful exit status",
                )
            )
            continue

        config_matches = config_re.findall(
            text
        )

        if not config_matches:
            problems.append(
                (
                    str(log_path),
                    "config path not found",
                )
            )
            continue

        # Final Config line at end of log.
        config_path = Path(
            config_matches[-1]
        )

        if not config_path.exists():
            problems.append(
                (
                    str(log_path),
                    f"config missing: {config_path}",
                )
            )
            continue

        with config_path.open() as f:
            cfg = json.load(f)

        classifiers = cfg[
            "ml"
        ][
            "classifiers"
        ]

        train_f1 = [
            float(x)
            for x in train_f1_re.findall(
                text
            )
        ]

        test_f1 = [
            float(x)
            for x in test_f1_re.findall(
                text
            )
        ]

        if (
            len(train_f1) != 3
            or len(test_f1) != 3
            or len(classifiers) != 3
        ):
            problems.append(
                (
                    str(log_path),
                    (
                        f"train F1={len(train_f1)}, "
                        f"test F1={len(test_f1)}, "
                        f"classifiers={len(classifiers)}"
                    ),
                )
            )
            continue

        name = cfg[
            "pipeline_name"
        ]

        location = None

        for candidate in LOCATIONS:
            if (
                f"_{candidate}_"
                in name
            ):
                location = candidate
                break

        if location is None:
            problems.append(
                (
                    str(log_path),
                    "location not recognized",
                )
            )
            continue

        attack_cfg = cfg[
            "attack"
        ]

        minimum = int(
            attack_cfg[
                "offset_distance_min"
            ]
        )

        maximum = int(
            attack_cfg[
                "offset_distance_max"
            ]
        )

        params = cfg.get(
            "template_parameters",
            {},
        )

        feature = params.get(
            "feature_set"
        )

        if feature is None:
            problems.append(
                (
                    str(log_path),
                    "feature_set missing",
                )
            )
            continue

        has_id = bool(
            params.get(
                "with_vehicle_id",
                False,
            )
        )

        for i, classifier in enumerate(
            classifiers
        ):

            rows.append(
                {
                    "attack": attack,
                    "location": location,
                    "range_min": minimum,
                    "range_max": maximum,
                    "range": (
                        f"{minimum}-{maximum}"
                    ),
                    "feature": feature,
                    "has_id": has_id,
                    "classifier": classifier,
                    "train_f1": train_f1[i],
                    "test_f1": test_f1[i],
                    "pipeline_name": name,
                    "config": str(
                        config_path
                    ),
                    "log": str(
                        log_path
                    ),
                }
            )


df = pd.DataFrame(rows)

expected_rows = 1152

print()
print("=" * 70)
print(
    f"Rows extracted: {len(df)}"
)
print(
    f"Expected rows:  {expected_rows}"
)

if problems:
    print()
    print(
        f"PROBLEMS: {len(problems)}"
    )

    for problem in problems[:30]:
        print(
            "  ",
            problem,
        )

if len(df) != expected_rows:
    raise RuntimeError(
        "Result count validation failed"
    )

duplicates = df.duplicated(
    subset=[
        "attack",
        "location",
        "range_min",
        "range_max",
        "feature",
        "has_id",
        "classifier",
    ]
)

if duplicates.any():
    raise RuntimeError(
        f"Duplicate result rows: "
        f"{int(duplicates.sum())}"
    )

expected_per_attack = (
    df.groupby(
        "attack"
    ).size()
)

print()
print(
    "Rows per attack:"
)
print(
    expected_per_attack
)

assert (
    expected_per_attack[
        "rpo"
    ]
    == 576
)

assert (
    expected_per_attack[
        "cpo"
    ]
    == 576
)

output_path = Path(
    "final_good_all_results.csv"
)

df.to_csv(
    output_path,
    index=False,
)

print()
print(
    "VALIDATION PASSED"
)
print(
    f"Saved: {output_path}"
)
print("=" * 70)


# ============================================================
# SUMMARY 1: MEAN F1 BY ATTACK / FEATURE / ID
# ============================================================

print()
print(
    "===== MEAN TEST F1 BY FEATURE ====="
)

summary = (
    df.groupby(
        [
            "attack",
            "has_id",
            "feature",
        ]
    )[
        "test_f1"
    ]
    .mean()
    .reset_index()
)

for attack in [
    "rpo",
    "cpo",
]:

    print()
    print(
        f"--- {attack.upper()} ---"
    )

    subset = summary[
        summary[
            "attack"
        ]
        == attack
    ]

    pivot = subset.pivot(
        index="feature",
        columns="has_id",
        values="test_f1",
    )

    pivot = pivot.rename(
        columns={
            False: "No_ID",
            True: "With_ID",
        }
    )

    ordered = [
        x
        for x in [
            "xy",
            "basic",
            "movement",
            "extended",
        ]
        if x in pivot.index
    ]

    print(
        pivot.loc[
            ordered
        ].round(6)
    )


# ============================================================
# SUMMARY 2: 5-15 m ONLY
# ============================================================

print()
print(
    "===== 5-15 m MEAN TEST F1 ====="
)

small = df[
    (df["range_min"] == 5)
    & (df["range_max"] == 15)
]

small_summary = (
    small.groupby(
        [
            "attack",
            "has_id",
            "feature",
        ]
    )[
        "test_f1"
    ]
    .mean()
    .reset_index()
)

for attack in [
    "rpo",
    "cpo",
]:

    print()
    print(
        f"--- {attack.upper()} 5-15 m ---"
    )

    subset = small_summary[
        small_summary[
            "attack"
        ]
        == attack
    ]

    pivot = subset.pivot(
        index="feature",
        columns="has_id",
        values="test_f1",
    )

    pivot = pivot.rename(
        columns={
            False: "No_ID",
            True: "With_ID",
        }
    )

    ordered = [
        x
        for x in [
            "xy",
            "basic",
            "movement",
            "extended",
        ]
        if x in pivot.index
    ]

    print(
        pivot.loc[
            ordered
        ].round(6)
    )


# ============================================================
# SUMMARY 3: CLASSIFIER PERFORMANCE
# ============================================================

print()
print(
    "===== MEAN TEST F1 BY CLASSIFIER ====="
)

print(
    df.groupby(
        [
            "attack",
            "classifier",
        ]
    )[
        "test_f1"
    ]
    .mean()
    .round(6)
)
