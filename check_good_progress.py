from pathlib import Path
import math
import re
import subprocess
import sys


JOBS = {
    "RPO": {
        "job_id": "18292702",
        "pattern": "good-rpo_18292702_*.out",
    },
    "CPO": {
        "job_id": "18292714",
        "pattern": "good-cpo_18292714_*.out",
    },
}

LOG_DIR = Path("slurm_final_good")

error_re = re.compile(
    r"Traceback \(most recent call last\):"
    r"|\bERROR:"
    r"|KeyError:"
    r"|ValueError:"
    r"|TypeError:"
    r"|RuntimeError:"
    r"|Exit status:\s*[1-9][0-9]*"
)

test_f1_re = re.compile(
    r"TEST SET RESULTS:.*?"
    r"F1 Score:\s*([0-9.eE+-]+)",
    re.DOTALL,
)

overall_ok = True


for attack, info in JOBS.items():

    print()
    print("=" * 70)
    print(f"{attack} CURRENT HEALTH")
    print("=" * 70)

    logs = sorted(
        LOG_DIR.glob(info["pattern"])
    )

    completed = []
    active_or_partial = []
    error_logs = []
    malformed = []
    all_f1 = []

    for path in logs:

        text = path.read_text(
            errors="replace"
        )

        if error_re.search(text):
            error_logs.append(path)
            continue

        if re.search(
            r"Exit status:\s*0",
            text,
        ):
            completed.append(path)

            f1_values = [
                float(value)
                for value
                in test_f1_re.findall(text)
            ]

            # Every config runs:
            # RF + DT + KNN
            if len(f1_values) != 3:
                malformed.append(
                    (
                        path,
                        f"expected 3 TEST F1 values, "
                        f"found {len(f1_values)}",
                    )
                )
                continue

            bad_values = [
                value
                for value in f1_values
                if (
                    not math.isfinite(value)
                    or value < 0.0
                    or value > 1.0
                )
            ]

            if bad_values:
                malformed.append(
                    (
                        path,
                        f"invalid F1 values: "
                        f"{bad_values}",
                    )
                )
                continue

            all_f1.extend(
                f1_values
            )

        else:
            active_or_partial.append(
                path
            )

    print(
        f"Log files created:       "
        f"{len(logs)}"
    )
    print(
        f"Completed successfully:  "
        f"{len(completed)}"
    )
    print(
        f"Running/partial logs:    "
        f"{len(active_or_partial)}"
    )
    print(
        f"Logs with errors:        "
        f"{len(error_logs)}"
    )
    print(
        f"Malformed completed:     "
        f"{len(malformed)}"
    )
    print(
        f"Valid TEST evaluations:  "
        f"{len(all_f1)}"
    )

    if all_f1:
        print(
            f"TEST F1 range so far:    "
            f"{min(all_f1):.6f} "
            f"to {max(all_f1):.6f}"
        )

        print(
            f"Mean TEST F1 so far:     "
            f"{sum(all_f1) / len(all_f1):.6f}"
        )

    if error_logs:

        overall_ok = False

        print()
        print("ERROR LOGS:")

        for path in error_logs[:20]:
            print(
                f"  {path}"
            )

    if malformed:

        overall_ok = False

        print()
        print(
            "MALFORMED COMPLETED LOGS:"
        )

        for path, reason in malformed[:20]:
            print(
                f"  {path}: {reason}"
            )

    # Check Slurm for already-known failed array tasks.
    try:

        result = subprocess.run(
            [
                "sacct",
                "-j",
                info["job_id"],
                "-X",
                "-n",
                "-P",
                "--format="
                "JobIDRaw,State,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        bad_states = []

        for line in result.stdout.splitlines():

            parts = line.split("|")

            if len(parts) < 3:
                continue

            job_id, state, exit_code = (
                parts[0],
                parts[1],
                parts[2],
            )

            # Only inspect actual array elements.
            if "_" not in job_id:
                continue

            state = state.split(
                "+"
            )[0]

            if state in {
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "BOOT_FAIL",
                "DEADLINE",
            }:
                bad_states.append(
                    (
                        job_id,
                        state,
                        exit_code,
                    )
                )

        print(
            f"Known failed Slurm tasks: "
            f"{len(bad_states)}"
        )

        if bad_states:

            overall_ok = False

            for item in bad_states[:20]:
                print(
                    "  ",
                    *item,
                )

    except Exception as exc:

        print(
            "WARNING: Could not query "
            f"sacct: {exc}"
        )


print()
print("=" * 70)

if overall_ok:

    print(
        "CURRENT GOOD RUNS LOOK "
        "HEALTHY SO FAR"
    )

    print(
        "All completed configs have "
        "3 valid TEST F1 results and "
        "no detected runtime errors."
    )

    print("=" * 70)

    sys.exit(0)

else:

    print(
        "PROBLEM DETECTED. DO NOT "
        "START HISTORICAL RUNS YET."
    )

    print("=" * 70)

    sys.exit(2)
