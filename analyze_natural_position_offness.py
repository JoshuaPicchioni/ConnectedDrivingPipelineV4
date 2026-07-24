from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ============================================================
# CONFIGURATION
# ============================================================

PARQUET_PATH = Path(
    "/scratch/picchioj/wyoming_april_2021/"
    "data/April_2021_Wyoming_Data_Fixed.parquet"
)

EARTH_RADIUS_M = 6_371_008.8

CENTERS = {
    "rock_springs": (
        41.538689,
        -109.319556,
    ),
    "laramie": (
        41.3100,
        -105.6000,
    ),
    "evanston": (
        41.2700,
        -110.9600,
    ),
}

REGION_RADIUS_M = 100_000.0

MIN_DT = 0.02
MAX_DT = 1.0
MAX_IMPLIED_SPEED_MPS = 100.0

THRESHOLDS_M = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    30.0,
]


# ============================================================
# VECTOR FUNCTIONS
# ============================================================

def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Vectorized Haversine distance in meters.
    Inputs are degrees.
    """

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return (
        2.0
        * EARTH_RADIUS_M
        * np.arcsin(
            np.sqrt(a)
        )
    )


def destination_point(
    lat_deg,
    lon_deg,
    heading_deg,
    distance_m,
):
    """
    Predict destination point from:
      previous latitude/longitude,
      heading,
      traveled distance.

    Heading convention:
      0 degrees   = North
      90 degrees  = East
      180 degrees = South
      270 degrees = West
    """

    lat1 = np.radians(
        lat_deg
    )

    lon1 = np.radians(
        lon_deg
    )

    theta = np.radians(
        heading_deg
    )

    delta = (
        distance_m
        / EARTH_RADIUS_M
    )

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    lat2 = np.arcsin(
        sin_lat1
        * cos_delta
        + cos_lat1
        * sin_delta
        * np.cos(theta)
    )

    lon2 = (
        lon1
        + np.arctan2(
            np.sin(theta)
            * sin_delta
            * cos_lat1,
            cos_delta
            - sin_lat1
            * np.sin(lat2),
        )
    )

    return (
        np.degrees(lat2),
        np.degrees(lon2),
    )


def initial_bearing_deg(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Observed movement bearing from
    previous position to current position.
    """

    lat1_rad = np.radians(
        lat1
    )

    lat2_rad = np.radians(
        lat2
    )

    dlon = np.radians(
        lon2 - lon1
    )

    x = (
        np.sin(dlon)
        * np.cos(lat2_rad)
    )

    y = (
        np.cos(lat1_rad)
        * np.sin(lat2_rad)
        - np.sin(lat1_rad)
        * np.cos(lat2_rad)
        * np.cos(dlon)
    )

    bearing = (
        np.degrees(
            np.arctan2(
                x,
                y,
            )
        )
        + 360.0
    ) % 360.0

    return bearing


def angular_difference_deg(
    a,
    b,
):
    """
    Smallest absolute difference
    between two headings.
    """

    return np.abs(
        (
            a
            - b
            + 180.0
        )
        % 360.0
        - 180.0
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def describe_values(
    values,
    scope,
    population,
    metric,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return None

    quantiles = np.quantile(
        values,
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999,
        ],
    )

    return {
        "scope": scope,
        "population": population,
        "metric": metric,
        "count": len(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "p01": quantiles[0],
        "p05": quantiles[1],
        "p10": quantiles[2],
        "p25": quantiles[3],
        "median": quantiles[4],
        "p75": quantiles[5],
        "p90": quantiles[6],
        "p95": quantiles[7],
        "p99": quantiles[8],
        "p999": quantiles[9],
        "max": np.max(values),
    }


def threshold_summary(
    values,
    scope,
    population,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    rows = []

    if len(values) == 0:
        return rows

    for threshold in THRESHOLDS_M:

        rows.append(
            {
                "scope": scope,
                "population": population,
                "threshold_m": threshold,
                "percent_le_threshold": (
                    np.mean(
                        values
                        <= threshold
                    )
                    * 100.0
                ),
                "percent_gt_threshold": (
                    np.mean(
                        values
                        > threshold
                    )
                    * 100.0
                ),
            }
        )

    return rows


def vehicle_summary(
    vehicle_codes,
    errors,
    mask,
    scope,
    population,
):
    subset = pd.DataFrame(
        {
            "vehicle": (
                vehicle_codes[
                    mask
                ]
            ),
            "error": (
                errors[
                    mask
                ]
            ),
        }
    )

    subset = subset[
        np.isfinite(
            subset[
                "error"
            ]
        )
    ]

    if subset.empty:
        return []

    grouped = subset.groupby(
        "vehicle",
        sort=False,
    )[
        "error"
    ]

    per_vehicle_mean = grouped.mean()
    per_vehicle_median = grouped.median()

    rows = []

    for (
        metric_name,
        values,
    ) in [
        (
            "per_vehicle_mean_prediction_error_m",
            per_vehicle_mean.to_numpy(),
        ),
        (
            "per_vehicle_median_prediction_error_m",
            per_vehicle_median.to_numpy(),
        ),
    ]:

        row = describe_values(
            values,
            scope,
            population,
            metric_name,
        )

        if row is not None:
            row[
                "vehicles"
            ] = len(values)

            rows.append(
                row
            )

    return rows


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading required Parquet columns..."
)

dataset = ds.dataset(
    str(PARQUET_PATH),
    format="parquet",
)

columns = [
    "metadata_generatedAt",
    "coreData_id",
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_speed",
    "coreData_heading",
]

table = dataset.to_table(
    columns=columns,
)

df = table.to_pandas()

print(
    f"Rows loaded: {len(df):,}"
)


# ============================================================
# CLEAN TYPES
# ============================================================

print(
    "Parsing timestamps and numeric columns..."
)

timestamps = pd.to_datetime(
    df[
        "metadata_generatedAt"
    ],
    errors="coerce",
    utc=True,
    format="mixed",
)

# Force nanosecond resolution explicitly.
# Pandas 3 may preserve microsecond resolution, so astype("int64")
# alone is not guaranteed to return nanoseconds.
df[
    "timestamp_ns"
] = timestamps.dt.as_unit(
    "ns"
).astype(
    "int64"
)

for column in [
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_speed",
    "coreData_heading",
]:

    df[
        column
    ] = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )

valid_base = (
    timestamps.notna()
    & df[
        "coreData_id"
    ].notna()
    & np.isfinite(
        df[
            "coreData_position_lat"
        ]
    )
    & np.isfinite(
        df[
            "coreData_position_long"
        ]
    )
    & np.isfinite(
        df[
            "coreData_speed"
        ]
    )
    & np.isfinite(
        df[
            "coreData_heading"
        ]
    )
)

df = df.loc[
    valid_base,
    [
        "coreData_id",
        "timestamp_ns",
        "coreData_position_lat",
        "coreData_position_long",
        "coreData_speed",
        "coreData_heading",
    ],
].copy()

print(
    f"Rows after base cleaning: "
    f"{len(df):,}"
)


# ============================================================
# CONVERT VEHICLE IDS TO INTEGER CODES
# ============================================================

print(
    "Encoding vehicle IDs..."
)

vehicle_codes, unique_ids = pd.factorize(
    df[
        "coreData_id"
    ],
    sort=False,
)

df[
    "vehicle_code"
] = vehicle_codes.astype(
    np.int64
)

df.drop(
    columns=[
        "coreData_id"
    ],
    inplace=True,
)

print(
    f"Unique vehicle IDs: "
    f"{len(unique_ids):,}"
)


# ============================================================
# SORT BY VEHICLE AND TIME
# ============================================================

print(
    "Sorting by vehicle and timestamp..."
)

df.sort_values(
    [
        "vehicle_code",
        "timestamp_ns",
    ],
    inplace=True,
    kind="mergesort",
)

df.reset_index(
    drop=True,
    inplace=True,
)


# ============================================================
# EXTRACT NUMPY ARRAYS
# ============================================================

vehicle = df[
    "vehicle_code"
].to_numpy(
    dtype=np.int64
)

timestamp_ns = df[
    "timestamp_ns"
].to_numpy(
    dtype=np.int64
)

lat = df[
    "coreData_position_lat"
].to_numpy(
    dtype=np.float64
)

lon = df[
    "coreData_position_long"
].to_numpy(
    dtype=np.float64
)

speed = df[
    "coreData_speed"
].to_numpy(
    dtype=np.float64
)

heading = df[
    "coreData_heading"
].to_numpy(
    dtype=np.float64
)

n = len(df)

del df
del table


# ============================================================
# PREVIOUS BSM VALUES
# ============================================================

same_vehicle = np.zeros(
    n,
    dtype=bool,
)

same_vehicle[
    1:
] = (
    vehicle[
        1:
    ]
    == vehicle[
        :-1
    ]
)

prev_lat = np.roll(
    lat,
    1,
)

prev_lon = np.roll(
    lon,
    1,
)

prev_speed = np.roll(
    speed,
    1,
)

prev_heading = np.roll(
    heading,
    1,
)

dt = np.full(
    n,
    np.nan,
    dtype=np.float64,
)

dt[
    1:
] = (
    timestamp_ns[
        1:
    ]
    - timestamp_ns[
        :-1
    ]
) / 1_000_000_000.0

dt[
    ~same_vehicle
] = np.nan


# ============================================================
# OBSERVED MOVEMENT
# ============================================================

print(
    "Calculating consecutive-BSM movement..."
)

step_distance_m = haversine_m(
    prev_lat,
    prev_lon,
    lat,
    lon,
)

observed_speed_mps = np.full(
    n,
    np.nan,
    dtype=np.float64,
)

np.divide(
    step_distance_m,
    dt,
    out=observed_speed_mps,
    where=(
        np.isfinite(dt)
        & (dt > 0.0)
    ),
)


# ============================================================
# CONTINUITY FILTER
# ============================================================

valid_interval = (
    same_vehicle
    & np.isfinite(dt)
    & (
        dt
        >= MIN_DT
    )
    & (
        dt
        <= MAX_DT
    )
    & np.isfinite(
        step_distance_m
    )
    & np.isfinite(
        observed_speed_mps
    )
    & (
        observed_speed_mps
        <= MAX_IMPLIED_SPEED_MPS
    )
    & np.isfinite(
        prev_speed
    )
    & np.isfinite(
        prev_heading
    )
)

print(
    f"Valid one-step intervals: "
    f"{valid_interval.sum():,}"
)


# ============================================================
# TWO-CONSECUTIVE-INTERVAL TRAJECTORY POPULATION
# ============================================================

previous_interval_valid = np.roll(
    valid_interval,
    1,
)

previous_interval_valid[
    0
] = False

trajectory_valid = (
    valid_interval
    & previous_interval_valid
    & same_vehicle
)

print(
    f"Two-interval trajectory-eligible rows: "
    f"{trajectory_valid.sum():,}"
)


# ============================================================
# EXPECTED DISTANCE
# ============================================================

expected_distance_m = (
    prev_speed
    * dt
)

distance_error_m = np.abs(
    step_distance_m
    - expected_distance_m
)


# ============================================================
# KINEMATIC POSITION PREDICTION
# ============================================================

print(
    "Calculating predicted positions..."
)

pred_lat, pred_lon = destination_point(
    prev_lat,
    prev_lon,
    prev_heading,
    expected_distance_m,
)

position_prediction_error_m = haversine_m(
    pred_lat,
    pred_lon,
    lat,
    lon,
)


# ============================================================
# SPEED ERROR
# ============================================================

speed_error_mps = np.abs(
    observed_speed_mps
    - prev_speed
)


# ============================================================
# HEADING ERROR
# ============================================================

observed_heading = initial_bearing_deg(
    prev_lat,
    prev_lon,
    lat,
    lon,
)

heading_error_deg = angular_difference_deg(
    observed_heading,
    prev_heading,
)


# ============================================================
# DEFINE SCOPES
# ============================================================

scopes = {
    "all_wyoming": np.ones(
        n,
        dtype=bool,
    )
}

print(
    "Calculating 100 km regional masks..."
)

for (
    name,
    (
        center_lat,
        center_lon,
    ),
) in CENTERS.items():

    current_distance = haversine_m(
        lat,
        lon,
        center_lat,
        center_lon,
    )

    previous_distance = haversine_m(
        prev_lat,
        prev_lon,
        center_lat,
        center_lon,
    )

    # Both endpoints of the interval
    # must be inside the 100 km region.
    scopes[
        name
    ] = (
        current_distance
        <= REGION_RADIUS_M
    ) & (
        previous_distance
        <= REGION_RADIUS_M
    )


# ============================================================
# GENERATE SUMMARIES
# ============================================================

metrics = {
    "position_prediction_error_m":
        position_prediction_error_m,

    "distance_error_m":
        distance_error_m,

    "step_distance_m":
        step_distance_m,

    "speed_error_mps":
        speed_error_mps,

    "heading_error_deg":
        heading_error_deg,
}

summary_rows = []
threshold_rows = []
vehicle_rows = []

for (
    scope_name,
    scope_mask,
) in scopes.items():

    for (
        population_name,
        population_mask,
    ) in [
        (
            "valid_one_step",
            valid_interval,
        ),
        (
            "trajectory_two_interval",
            trajectory_valid,
        ),
    ]:

        mask = (
            scope_mask
            & population_mask
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"{scope_name.upper()} | "
            f"{population_name}"
        )

        print(
            f"Intervals: "
            f"{mask.sum():,}"
        )

        for (
            metric_name,
            values,
        ) in metrics.items():

            row = describe_values(
                values[
                    mask
                ],
                scope_name,
                population_name,
                metric_name,
            )

            if row is not None:
                summary_rows.append(
                    row
                )

        error_values = (
            position_prediction_error_m[
                mask
            ]
        )

        threshold_rows.extend(
            threshold_summary(
                error_values,
                scope_name,
                population_name,
            )
        )

        vehicle_rows.extend(
            vehicle_summary(
                vehicle,
                position_prediction_error_m,
                mask,
                scope_name,
                population_name,
            )
        )

        # Print the main statistic directly.
        main = describe_values(
            error_values,
            scope_name,
            population_name,
            "position_prediction_error_m",
        )

        if main is not None:

            print()
            print(
                "NATURAL POSITION-PREDICTION "
                "ERROR (meters)"
            )

            print(
                f"Mean:    "
                f"{main['mean']:.4f}"
            )

            print(
                f"Median:  "
                f"{main['median']:.4f}"
            )

            print(
                f"P75:     "
                f"{main['p75']:.4f}"
            )

            print(
                f"P90:     "
                f"{main['p90']:.4f}"
            )

            print(
                f"P95:     "
                f"{main['p95']:.4f}"
            )

            print(
                f"P99:     "
                f"{main['p99']:.4f}"
            )

            print(
                f"P99.9:   "
                f"{main['p999']:.4f}"
            )

            print(
                f"Max:     "
                f"{main['max']:.4f}"
            )


# ============================================================
# SAVE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

threshold_df = pd.DataFrame(
    threshold_rows
)

vehicle_df = pd.DataFrame(
    vehicle_rows
)

summary_df.to_csv(
    "natural_offness_summary.csv",
    index=False,
)

threshold_df.to_csv(
    "natural_offness_thresholds.csv",
    index=False,
)

vehicle_df.to_csv(
    "natural_offness_vehicle_summary.csv",
    index=False,
)


# ============================================================
# PRINT MAIN TRAJECTORY SUMMARY
# ============================================================

print()
print(
    "=" * 90
)

print(
    "FINAL TRAJECTORY-POPULATION "
    "POSITION ERROR SUMMARY"
)

print(
    "=" * 90
)

main_table = summary_df[
    (
        summary_df[
            "population"
        ]
        == "trajectory_two_interval"
    )
    & (
        summary_df[
            "metric"
        ]
        == "position_prediction_error_m"
    )
][
    [
        "scope",
        "count",
        "mean",
        "median",
        "p75",
        "p90",
        "p95",
        "p99",
        "p999",
        "max",
    ]
]

print(
    main_table.to_string(
        index=False
    )
)


print()
print(
    "=" * 90
)

print(
    "PERCENTAGE OF NATURAL "
    "PREDICTION ERRORS <= THRESHOLD"
)

print(
    "=" * 90
)

threshold_table = threshold_df[
    threshold_df[
        "population"
    ]
    == "trajectory_two_interval"
].pivot(
    index="scope",
    columns="threshold_m",
    values="percent_le_threshold",
)

print(
    threshold_table.round(
        3
    ).to_string()
)


print()
print(
    "Saved:"
)

print(
    "  natural_offness_summary.csv"
)

print(
    "  natural_offness_thresholds.csv"
)

print(
    "  natural_offness_vehicle_summary.csv"
)

print()
print(
    "ANALYSIS COMPLETE"
)
