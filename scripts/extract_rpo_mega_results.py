from __future__ import annotations

import csv
import re
from pathlib import Path


LOG_DIRS = [
    ("core", Path("rpo_mega_core_logs/job_18189142")),
    ("id", Path("rpo_mega_id_logs/job_18189144")),
]

OUTPUT = Path("rpo_mega_all_results.csv")

FILENAME_RE = re.compile(
    r"rpo_mega_"
    r"(rock_springs|laramie|evanston)_"
    r"(\d+)_(\d+)_"
    r"(xy|basic|movement|extended)"
    r"(_id)?\.out$"
)

CLASSIFIER_RE = re.compile(
    r"CLASSIFIER:\s*"
    r"(RandomForestClassifier|DecisionTreeClassifier|KNeighborsClassifier)"
)

METRIC_PATTERNS = {
    "accuracy": re.compile(r"Accuracy:\s*([0-9.]+)"),
    "precision": re.compile(r"Precision:\s*([0-9.]+)"),
    "recall": re.compile(r"Recall:\s*([0-9.]+)"),
    "f1": re.compile(r"F1 Score:\s*([0-9.]+)"),
    "specificity": re.compile(r"Specificity:\s*([0-9.]+)"),
}


def parse_log(path: Path, priority: str) -> list[dict]:
    filename_match = FILENAME_RE.match(path.name)

    if not filename_match:
        print(f"SKIPPING unrecognized filename: {path.name}")
        return []

    location, range_min, range_max, feature_set, id_suffix = (
        filename_match.groups()
    )

    current_classifier = None
    in_test = False
    metrics = {}
    rows = []

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in text.splitlines():

        classifier_match = CLASSIFIER_RE.search(line)

        if classifier_match:
            current_classifier = classifier_match.group(1)

        if "TRAIN SET RESULTS:" in line:
            in_test = False
            continue

        if "TEST SET RESULTS:" in line:
            in_test = True
            metrics = {}
            continue

        if not in_test:
            continue

        for metric_name, pattern in METRIC_PATTERNS.items():
            match = pattern.search(line)

            if match:
                metrics[metric_name] = float(match.group(1))

        if "specificity" in metrics:

            if current_classifier is None:
                raise RuntimeError(
                    f"Test results found without classifier in {path}"
                )

            rows.append(
                {
                    "priority": priority,
                    "location": location,
                    "range_min": int(range_min),
                    "range_max": int(range_max),
                    "range": f"{range_min}-{range_max}",
                    "feature_set": feature_set,
                    "has_id": id_suffix is not None,
                    "classifier": current_classifier,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "specificity": metrics.get("specificity"),
                    "log_file": str(path),
                }
            )

            in_test = False
            metrics = {}

    return rows


def main() -> None:

    all_rows = []

    for priority, directory in LOG_DIRS:

        if not directory.exists():
            print(f"ERROR: missing log directory: {directory}")
            continue

        logs = sorted(directory.glob("*.out"))

        print(
            f"{priority}: found {len(logs)} log files"
        )

        for log in logs:
            all_rows.extend(
                parse_log(
                    log,
                    priority,
                )
            )

    fieldnames = [
        "priority",
        "location",
        "range_min",
        "range_max",
        "range",
        "feature_set",
        "has_id",
        "classifier",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "log_file",
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("RESULT EXTRACTION COMPLETE")
    print("--------------------------")
    print(f"Rows extracted: {len(all_rows)}")
    print("Expected rows:  576")
    print(f"Output:         {OUTPUT}")

    if len(all_rows) == 576:
        print()
        print("VALIDATION PASSED: all 576 results found.")
    else:
        print()
        print(
            "WARNING: Expected 576 results. "
            "Inspect logs before using the CSV."
        )


if __name__ == "__main__":
    main()
