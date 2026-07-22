import os
from pathlib import Path

import pandas as pd

SOURCE = Path(
    os.path.expandvars(
        "$SCRATCH/wyoming_april_2021/raw/April_2021_Wyoming_Data.csv"
    )
)

OUTPUT = Path(
    os.path.expandvars(
        "$SCRATCH/wyoming_april_2021/data/April_2021_Wyoming_Data_Fixed.parquet"
    )
)

CHUNK_SIZE = 500_000
EXPECTED_ROWS = 13_318_200
EXPECTED_COLUMNS = 69

# Rename Socrata-lowercased headers to names used by the pipeline.
# Columns not listed here retain their original names.
RENAME_MAP = {
    "metadata_generatedat": "metadata_generatedAt",
    "metadata_receivedat": "metadata_receivedAt",

    "coredata_msgcnt": "coreData_msgCnt",
    "coredata_id": "coreData_id",
    "coredata_secmark": "coreData_secMark",
    "coredata_position_lat": "coreData_position_lat",
    "coredata_position_long": "coreData_position_long",
    "coredata_accuracy_semimajor": "coreData_accuracy_semiMajor",
    "coredata_accuracy_semiminor": "coreData_accuracy_semiMinor",
    "coredata_elevation": "coreData_elevation",
    "coredata_accelset_accelyaw": "coreData_accelset_accelYaw",
    "coredata_speed": "coreData_speed",
    "coredata_heading": "coreData_heading",
    "coredata_position": "coreData_position",
}

INTEGER_COLUMNS = [
    "metadata_serialid_bundlesize",
    "metadata_serialid_bundleid",
    "metadata_serialid_recordid",
    "metadata_serialid_serialnumber",
    "coreData_msgCnt",
    "coreData_secMark",
]

FLOAT_COLUMNS = [
    "coreData_position_lat",
    "coreData_position_long",
    "coreData_accuracy_semiMajor",
    "coreData_accuracy_semiMinor",
    "coreData_elevation",
    "coreData_accelset_accelYaw",
    "coreData_speed",
    "coreData_heading",
]

if not SOURCE.exists():
    raise FileNotFoundError(f"Source CSV not found: {SOURCE}")

if OUTPUT.exists():
    raise RuntimeError(f"Output already exists: {OUTPUT}")

OUTPUT.mkdir(parents=True)

total_rows = 0
partition_count = 0

print(f"Source: {SOURCE}", flush=True)
print(f"Output: {OUTPUT}", flush=True)
print(f"Chunk size: {CHUNK_SIZE:,}", flush=True)

for i, chunk in enumerate(
    pd.read_csv(
        SOURCE,
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
):
    # Rename columns instead of duplicating them.
    chunk = chunk.rename(columns=RENAME_MAP)

    if len(chunk.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Column count mismatch in partition {i}: "
            f"expected {EXPECTED_COLUMNS}, got {len(chunk.columns)}"
        )

    for col in INTEGER_COLUMNS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            ).astype("Int64")

    for col in FLOAT_COLUMNS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(
                chunk[col], errors="coerce"
            ).astype("float64")

    output_file = OUTPUT / f"part.{i:04d}.parquet"

    chunk.to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    total_rows += len(chunk)
    partition_count += 1

    print(
        f"Partition {i:04d}: {len(chunk):,} rows | "
        f"Total: {total_rows:,}",
        flush=True,
    )

print()
print("=" * 60)
print("CONVERSION COMPLETE")
print(f"Total rows: {total_rows:,}")
print(f"Partitions: {partition_count}")
print(f"Columns: {EXPECTED_COLUMNS}")
print("=" * 60)

if total_rows != EXPECTED_ROWS:
    raise RuntimeError(
        f"ROW COUNT MISMATCH: expected {EXPECTED_ROWS:,}, "
        f"got {total_rows:,}"
    )

if partition_count != 27:
    raise RuntimeError(
        f"PARTITION COUNT MISMATCH: expected 27, got {partition_count}"
    )

print("VALIDATION PASSED", flush=True)
