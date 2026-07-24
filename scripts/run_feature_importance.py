from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from MachineLearning.DaskPipelineRunner import DaskPipelineRunner


def atomic_csv_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(destination)


def atomic_json_write(
    data: dict,
    destination: Path,
) -> None:
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
        )
    )

    temporary.replace(destination)


def deterministic_stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    maximum_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if len(X) <= maximum_rows:
        return X, y

    indices = np.arange(len(X))

    stratification = (
        y
        if y.nunique(dropna=False) > 1
        else None
    )

    selected, _ = train_test_split(
        indices,
        train_size=maximum_rows,
        random_state=seed,
        shuffle=True,
        stratify=stratification,
    )

    selected = np.sort(selected)

    return (
        X.iloc[selected].reset_index(drop=True),
        y.iloc[selected].reset_index(drop=True),
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/run_feature_importance.py "
            "<config.json>",
            file=sys.stderr,
        )
        return 2

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config does not exist: {config_path}"
        )

    with config_path.open() as handle:
        config = json.load(handle)

    pipeline_name = config["pipeline_name"]

    fi_config = config.get(
        "feature_importance",
        {},
    )

    output_directory = Path(
        fi_config.get(
            "results_dir",
            "results/feature_importance_combined",
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / f"{pipeline_name}.csv"
    )

    json_path = (
        output_directory
        / f"{pipeline_name}.json"
    )

    # Atomic writes mean an existing CSV represents a completed run.
    if csv_path.exists() and json_path.exists():
        print(
            f"SKIP: completed outputs already exist for "
            f"{pipeline_name}"
        )
        return 0

    maximum_rows = int(
        os.environ.get(
            "FI_MAX_TEST_ROWS",
            "5000",
        )
    )

    repetitions = int(
        os.environ.get(
            "FI_N_REPEATS",
            "10",
        )
    )

    random_seed = int(
        os.environ.get(
            "FI_RANDOM_SEED",
            "42",
        )
    )

    configured_features = list(
        config["ml"]["features"]
    )

    print("=" * 72)
    print("FEATURE IMPORTANCE")
    print(f"Pipeline:       {pipeline_name}")
    print(f"Config:         {config_path}")
    print(f"Features:       {len(configured_features)}")
    print(f"ID included:    {'coreData_id' in configured_features}")
    print(f"Maximum rows:   {maximum_rows}")
    print(f"Repetitions:    {repetitions}")
    print(f"Random seed:    {random_seed}")
    print("=" * 72)

    start_time = time.time()

    runner = DaskPipelineRunner.from_config(
        str(config_path)
    )

    results, pipeline_metadata = (
        runner.run_with_metadata()
    )

    scorer = make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0,
    )

    output_rows: list[dict] = []
    model_summaries: list[dict] = []

    for (
        model_wrapper,
        train_metrics,
        test_metrics,
    ) in results:
        classifier = model_wrapper.classifier
        classifier_name = (
            classifier.__class__.__name__
        )

        X_full = model_wrapper.test_X
        y_full = model_wrapper.test_Y

        if not isinstance(X_full, pd.DataFrame):
            X_full = pd.DataFrame(
                X_full,
                columns=configured_features,
            )

        if not isinstance(y_full, pd.Series):
            y_full = pd.Series(
                np.asarray(y_full).reshape(-1)
            )

        missing_features = [
            feature
            for feature in configured_features
            if feature not in X_full.columns
        ]

        if missing_features:
            raise RuntimeError(
                f"{classifier_name} is missing features: "
                f"{missing_features}"
            )

        X_full = X_full[
            configured_features
        ].copy()

        X_sample, y_sample = (
            deterministic_stratified_sample(
                X_full,
                y_full,
                maximum_rows=maximum_rows,
                seed=random_seed,
            )
        )

        class_counts = (
            y_sample.value_counts()
            .sort_index()
            .to_dict()
        )

        print()
        print("-" * 72)
        print(f"Classifier:       {classifier_name}")
        print(f"Full test rows:   {len(X_full):,}")
        print(f"Sample rows:      {len(X_sample):,}")
        print(f"Sample classes:   {class_counts}")
        print(f"Full test F1:     {float(test_metrics[3]):.6f}")

        baseline_predictions = classifier.predict(
            X_sample
        )

        sample_baseline_f1 = f1_score(
            y_sample,
            baseline_predictions,
            pos_label=1,
            zero_division=0,
        )

        print(
            f"Sample baseline F1: "
            f"{sample_baseline_f1:.6f}"
        )

        importance_start = time.time()

        permutation = permutation_importance(
            classifier,
            X_sample,
            y_sample,
            scoring=scorer,
            n_repeats=repetitions,
            random_state=random_seed,
            n_jobs=1,
        )

        importance_seconds = (
            time.time() - importance_start
        )

        native_importances = getattr(
            classifier,
            "feature_importances_",
            None,
        )

        model_rows: list[dict] = []

        for index, feature in enumerate(
            configured_features
        ):
            repeated_values = permutation.importances[
                index
            ]

            native_value = None

            if native_importances is not None:
                native_value = float(
                    native_importances[index]
                )

            row = {
                "pipeline_name": pipeline_name,
                "source_config": str(config_path),
                "attack_type": config["attack"]["type"],
                "range_min": float(
                    config["attack"][
                        "offset_distance_min"
                    ]
                ),
                "range_max": float(
                    config["attack"][
                        "offset_distance_max"
                    ]
                ),
                "region": config.get(
                    "template_parameters",
                    {},
                ).get(
                    "region",
                    pipeline_name,
                ),
                "classifier": classifier_name,
                "feature": feature,
                "feature_index": index,
                "full_test_f1": float(
                    test_metrics[3]
                ),
                "sample_baseline_f1": float(
                    sample_baseline_f1
                ),
                "sample_rows": int(
                    len(X_sample)
                ),
                "sample_clean_rows": int(
                    class_counts.get(0, 0)
                ),
                "sample_attacker_rows": int(
                    class_counts.get(1, 0)
                ),
                "permutation_importance_mean": float(
                    permutation.importances_mean[index]
                ),
                "permutation_importance_std": float(
                    permutation.importances_std[index]
                ),
                "permutation_importance_min": float(
                    repeated_values.min()
                ),
                "permutation_importance_max": float(
                    repeated_values.max()
                ),
                "native_tree_importance": (
                    native_value
                ),
                "n_repeats": repetitions,
                "random_seed": random_seed,
                "importance_seconds": float(
                    importance_seconds
                ),
            }

            model_rows.append(row)

        model_dataframe = pd.DataFrame(
            model_rows
        )

        model_dataframe[
            "permutation_rank"
        ] = (
            model_dataframe[
                "permutation_importance_mean"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        model_dataframe = model_dataframe.sort_values(
            [
                "permutation_rank",
                "feature_index",
            ]
        )

        output_rows.extend(
            model_dataframe.to_dict(
                orient="records"
            )
        )

        print()
        print("Top permutation importances:")

        for _, row in model_dataframe.head(8).iterrows():
            print(
                f"  {int(row['permutation_rank']):2d}. "
                f"{row['feature']:<44} "
                f"{row['permutation_importance_mean']:+.6f} "
                f"± {row['permutation_importance_std']:.6f}"
            )

        model_summaries.append({
            "classifier": classifier_name,
            "full_test_f1": float(
                test_metrics[3]
            ),
            "sample_baseline_f1": float(
                sample_baseline_f1
            ),
            "sample_rows": int(
                len(X_sample)
            ),
            "class_counts": {
                str(key): int(value)
                for key, value
                in class_counts.items()
            },
            "importance_seconds": float(
                importance_seconds
            ),
        })

    final_dataframe = pd.DataFrame(
        output_rows
    )

    expected_rows = (
        len(configured_features)
        * len(results)
    )

    if len(final_dataframe) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} output rows, "
            f"found {len(final_dataframe)}"
        )

    final_dataframe = final_dataframe.sort_values(
        [
            "classifier",
            "permutation_rank",
            "feature_index",
        ]
    )

    total_seconds = time.time() - start_time

    summary = {
        "pipeline_name": pipeline_name,
        "config_path": str(config_path),
        "feature_count": len(
            configured_features
        ),
        "features": configured_features,
        "vehicle_id_included": (
            "coreData_id"
            in configured_features
        ),
        "classifier_count": len(results),
        "maximum_test_rows": maximum_rows,
        "n_repeats": repetitions,
        "random_seed": random_seed,
        "total_seconds": total_seconds,
        "models": model_summaries,
        "pipeline_vehicle_metadata": {
            "total_unique_vehicle_ids": (
                pipeline_metadata.get(
                    "total_unique_vehicle_ids"
                )
            ),
            "train_vehicle_stats": (
                pipeline_metadata.get(
                    "train_vehicle_stats"
                )
            ),
            "test_vehicle_stats": (
                pipeline_metadata.get(
                    "test_vehicle_stats"
                )
            ),
        },
    }

    atomic_csv_write(
        final_dataframe,
        csv_path,
    )

    atomic_json_write(
        summary,
        json_path,
    )

    print()
    print("=" * 72)
    print("FEATURE IMPORTANCE COMPLETE")
    print(f"CSV:          {csv_path}")
    print(f"JSON:         {json_path}")
    print(f"Output rows:  {len(final_dataframe)}")
    print(f"Total time:   {total_seconds:.2f}s")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
