# Patch to add skip logic

SKIP_ATTACKS = """
# SKIP LOGIC: Heavy attacks on large datasets (2026-02-24)
# Reason: swap_rand and override_* attacks cause Dask deadlocks on 100km/200km data
# These attack types require row-by-row operations that don't parallelize well
# with 3.4M+ row datasets, causing memory issues and worker deadlocks.
#
# Skipped combinations:
#   - swap_rand on 100km (3.4M rows)
#   - swap_rand on 200km (even larger)
#   - override_const on 100km/200km
#   - override_rand on 100km/200km
#
# Result: 162 - 36 = 126 pipelines will run
# (2km still gets all 6 attacks = 54 pipelines)
# (100km/200km get 3 attacks each = 54 + 18 = 72 pipelines)

SKIP_COMBINATIONS = {
    "swap_rand": ["100km", "200km"],
    "override_const": ["100km", "200km"],
    "override_rand": ["100km", "200km"],
}

def should_skip(features_name, radius_name, attack_name):
    if attack_name in SKIP_COMBINATIONS:
        if radius_name in SKIP_COMBINATIONS[attack_name]:
            return True
    return False
"""
print(SKIP_ATTACKS)
