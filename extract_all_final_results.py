from pathlib import Path
import json
import re
import pandas as pd


RUNS = [
    {
        "experiment_group": "good",
        "attack": "rpo",
        "job_id": "18292702",
        "log_dir": "slurm_final_good",
        "pattern": "good-rpo_18292702_*.out",
    },
    {
        "experiment_group": "good",
        "attack": "cpo",
        "job_id": "18292714",
        "log_dir": "slurm_final_good",
        "pattern": "good-cpo_18292714_*.out",
    },
    {
        "experiment_group": "historical",
        "attack": "rpo",
        "job_id": "18301162",
        "log_dir": "slurm_final_bad",
        "pattern": "bad-rpo_18301162_*.out",
    },
    {
        "experiment_group": "historical",
        "attack": "cpo",
        "job_id": "18301163",
        "log_dir": "slurm_final_bad",
        "pattern": "bad-cpo_18301163_*.out",
    },
    {
        "experiment_group": "small_range",
        "attack": "rpo",
        "job_id": "18307990",
        "log_dir": "slurm_final_good_small",
        "pattern": "small-rpo_18307990_*.out",
    },
    {
        "experiment_group": "small_range",
        "attack": "cpo",
        "job_id": "18307991",
        "log_dir": "slurm_final_good_small",
        "pattern": "small-cpo_18307991_*.out",
    },
]


LOCATIONS = [
    "rock_springs",
    "laramie",
    "evanston",
]


config_re = re.compile(
    r"Config:\s*(\S+\.json)"
)

result_block_re = re.compile(
    r"(TRAIN|TEST) SET RESULTS:\s*"
    r"(.*?)"
    r"(?=(?:TRAIN|TEST) SET RESULTS:|$)",
    re.DOTALL,
)

metric_patterns = {
    "accuracy": re.compile(
        r"Accuracy:\s*([0-9.eE+-]+)"
    ),
    "precision": re.compile(
        r"Precision:\s*([0-9.eE+-]+)"
    ),
    "recall": re.compile(
        r"Recall:\s*([0-9.eE+-]+)"
    ),
    "f1": re.compile(
        r"F1 Score:\s*([0-9.eE+-]+)"
    ),
}


def extract_metrics(block):
    result = {}

    for name, pattern in metric_patterns.items():
        match = pattern.search(block)

        result[name] = (
            float(match.group(1))
            if match
            else None
        )

    return result


rows = []
problems = []


for run in RUNS:

    log_dir = Path(
        run["log_dir"]
    )

    logs = sorted(
        log_dir.glob(
            run["pattern"]
        )
    )

    successful_logs = 0

    print()
    print("=" * 70)
    print(
        f"{run['experiment_group'].upper()} "
        f"{run['attack'].upper()}"
    )
    print(
        f"Logs found: {len(logs)}"
    )

    for log_path in logs:

        text = log_path.read_text(
            errors="replace"
        )

        # Skip jobs that are still running.
        exit_match = re.search(
            r"Exit status:\s*(\d+)",
            text,
        )

        if not exit_match:
            continue

        if int(
            exit_match.group(1)
        ) != 0:

            problems.append(
                (
                    str(log_path),
                    "nonzero exit status",
                )
            )

            continue

        successful_logs += 1

        config_matches = (
            config_re.findall(
                text
            )
        )

        if not config_matches:

            problems.append(
                (
                    str(log_path),
                    "config path missing",
                )
            )

            continue

        config_path = Path(
            config_matches[-1]
        )

        if not config_path.exists():

            problems.append(
                (
                    str(log_path),
                    (
                        "config file does not exist: "
                        f"{config_path}"
                    ),
                )
            )

            continue

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            cfg = json.load(
                handle
            )

        pipeline_name = cfg.get(
            "pipeline_name",
            config_path.stem,
        )

        location = None

        for candidate in LOCATIONS:

            if (
                f"_{candidate}_"
                in pipeline_name
            ):
                location = candidate
                break

        if location is None:

            problems.append(
                (
                    str(log_path),
                    "could not determine location",
                )
            )

            continue

        attack_cfg = cfg.get(
            "attack",
            {},
        )

        range_min = attack_cfg.get(
            "offset_distance_min"
        )

        range_max = attack_cfg.get(
            "offset_distance_max"
        )

        params = cfg.get(
            "template_parameters",
            {},
        )

        feature_set = params.get(
            "feature_set"
        )

        features = (
            cfg.get(
                "ml",
                {},
            ).get(
                "features",
                [],
            )
        )

        if feature_set is None:

            # Fallback inference from pipeline name.
            name_lower = (
                pipeline_name.lower()
            )

            if "extended" in name_lower:
                feature_set = "extended"
            elif "movement" in name_lower:
                feature_set = "movement"
            elif "basic" in name_lower:
                feature_set = "basic"
            elif "_xy" in name_lower:
                feature_set = "xy"
            else:
                feature_set = "unknown"

        has_id = (
            "coreData_id"
            in features
        )

        has_all3ids = (
            "coreData_id"
            in features
            and "coreData_msgCnt"
            in features
            and "metadata_receivedAt"
            in features
        )

        classifiers = (
            cfg.get(
                "ml",
                {},
            ).get(
                "classifiers",
                [],
            )
        )

        blocks = (
            result_block_re.findall(
                text
            )
        )

        train_results = []
        test_results = []

        for kind, block in blocks:

            metrics = extract_metrics(
                block
            )

            if kind == "TRAIN":
                train_results.append(
                    metrics
                )
            else:
                test_results.append(
                    metrics
                )

        if (
            len(train_results)
            != len(classifiers)
            or len(test_results)
            != len(classifiers)
        ):

            problems.append(
                (
                    str(log_path),
                    (
                        f"classifiers={len(classifiers)}, "
                        f"train blocks="
                        f"{len(train_results)}, "
                        f"test blocks="
                        f"{len(test_results)}"
                    ),
                )
            )

            continue

        for i, classifier in enumerate(
            classifiers
        ):

            train = train_results[i]
            test = test_results[i]

            rows.append(
                {
                    "experiment_group":
                        run[
                            "experiment_group"
                        ],

                    "attack":
                        run[
                            "attack"
                        ],

                    "location":
                        location,

                    "range_min":
                        range_min,

                    "range_max":
                        range_max,

                    "range":
                        (
                            f"{range_min}-"
                            f"{range_max}"
                        ),

                    "feature_set":
                        feature_set,

                    "has_id":
                        has_id,

                    "has_all3ids":
                        has_all3ids,

                    "classifier":
                        classifier,

                    "train_accuracy":
                        train[
                            "accuracy"
                        ],

                    "train_precision":
                        train[
                            "precision"
                        ],

                    "train_recall":
                        train[
                            "recall"
                        ],

                    "train_f1":
                        train[
                            "f1"
                        ],

                    "test_accuracy":
                        test[
                            "accuracy"
                        ],

                    "test_precision":
                        test[
                            "precision"
                        ],

                    "test_recall":
                        test[
                            "recall"
                        ],

                    "test_f1":
                        test[
                            "f1"
                        ],

                    "pipeline_name":
                        pipeline_name,

                    "config":
                        str(
                            config_path
                        ),

                    "log":
                        str(
                            log_path
                        ),
                }
            )

    print(
        "Successfully completed "
        f"config logs: "
        f"{successful_logs}"
    )


df = pd.DataFrame(
    rows
)

output = Path(
    "all_final_experiment_results.csv"
)

df.to_csv(
    output,
    index=False,
)


print()
print("=" * 80)
print("EXTRACTION SUMMARY")
print("=" * 80)

print(
    f"Total classifier rows: "
    f"{len(df)}"
)

if not df.empty:

    summary = (
        df.groupby(
            [
                "experiment_group",
                "attack",
            ]
        )
        .size()
    )

    print()
    print(summary)


print()
print(
    f"Problems detected: "
    f"{len(problems)}"
)

for problem in problems[:50]:

    print(
        "  ",
        problem,
    )


print()
print(
    f"Saved: {output}"
)


# Expected final count once EVERYTHING is done:
#
# GOOD:
#   576 RPO + 576 CPO = 1,152
#
# HISTORICAL:
#   648 RPO + 648 CPO = 1,296
#
# SMALL RANGE:
#   360 RPO + 360 CPO = 720
#
# GRAND TOTAL:
#   3,168 classifier evaluations

if len(df) == 3168:

    print()
    print(
        "FULL FINAL MATRIX COMPLETE: "
        "3,168 / 3,168 evaluations"
    )
