#!/usr/bin/env python3
"""
Wrapper script to run pipelines with skip logic for heavy attacks.
Skips swap_rand/override_* on 100km/200km due to Dask deadlocks.
Also resumes from existing progress.

Created: 2026-02-24
Reason: swap_rand and override_* attacks cause Dask deadlocks on 3.4M+ row datasets
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Import the original pipeline runner
from run_162_pipelines import (
    ATTACK_CONFIGS, FEATURE_SETS, RADII_CONFIGS,
    generate_pipeline_config, run_pipeline
)

# SKIP LOGIC: Heavy attacks on large datasets cause Dask deadlocks
SKIP_COMBINATIONS = {
    "swap_rand": ["100km", "200km"],
    "override_const": ["100km", "200km"],
    "override_rand": ["100km", "200km"],
}

def should_skip(radius_name, attack_name):
    """Return True if this radius+attack combo should be skipped."""
    if attack_name in SKIP_COMBINATIONS:
        if radius_name in SKIP_COMBINATIONS[attack_name]:
            return True
    return False

def generate_filtered_configs():
    """Generate configs, filtering out heavy attack combinations."""
    configs = []
    skipped = []
    for features_name in FEATURE_SETS.keys():
        for radius_name in RADII_CONFIGS.keys():
            for attack_name in ATTACK_CONFIGS.keys():
                if should_skip(radius_name, attack_name):
                    skipped.append(f"{features_name}_{radius_name}_{attack_name}")
                else:
                    configs.append(generate_pipeline_config(features_name, radius_name, attack_name))
    return configs, skipped

def main():
    results_dir = "/var/www/static/pipeline-results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    configs, skipped = generate_filtered_configs()
    
    print("=" * 70)
    print("CONNECTED DRIVING PIPELINE RUNNER (WITH SKIP LOGIC)")
    print("=" * 70)
    print(f"\n⚠️  SKIPPING {len(skipped)} heavy attack combinations:")
    print("    - swap_rand on 100km/200km")
    print("    - override_const on 100km/200km")
    print("    - override_rand on 100km/200km")
    print("    Reason: Dask deadlock on large datasets (3.4M+ rows)\n")
    
    print(f"Pipelines to run: {len(configs)}")
    print(f"9 Feature Sets: {list(FEATURE_SETS.keys())}")
    print(f"3 Radii: {list(RADII_CONFIGS.keys())}")
    print(f"Active Attacks: rand_offset, const_offset, const_offset_per_id")
    print(f"                (+ all 6 attacks for 2km radius)")
    
    # Load existing progress
    progress_file = Path(results_dir) / "progress_162.json"
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        already_done = {r["name"] for r in progress["results"]}
        print(f"\n📊 Resuming: {len(already_done)} already completed")
    else:
        progress = {"total": len(configs), "completed": 0, "failed": 0, "results": [], 
                   "started_at": datetime.now().isoformat()}
        already_done = set()
    
    # Update total
    progress["total"] = len(configs)
    print(f"Remaining: {len(configs) - len(already_done)}\n")
    
    # Run pipelines
    for i, config in enumerate(configs):
        pname = config["pipeline_name"]
        
        if pname in already_done:
            print(f"[{i+1}/{len(configs)}] ⏭️  {pname} (done)")
            continue
        
        print(f"\n[{i+1}/{len(configs)}] Running {pname}")
        success, elapsed = run_pipeline(config, results_dir)
        
        if success:
            progress["completed"] += 1
        else:
            progress["failed"] += 1
        
        progress["results"].append({"name": pname, "success": success, "elapsed": elapsed})
        
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: {progress["completed"]}/{progress["total"]} succeeded, {progress["failed"]} failed")
    print("=" * 70)

if __name__ == "__main__":
    main()
