#!/usr/bin/env python3
"""
Autonomous Pipeline Executor for ConnectedDrivingPipelineV4
Executes all 36 pipelines sequentially with log auditing and Slack updates

Created: 2026-02-23
"""

import subprocess
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4/production_configs_v2"
PROJECT_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4"
LOG_DIR = f"{PROJECT_DIR}/logs"
RESULTS_DIR = f"{PROJECT_DIR}/results/matrix"
PROGRESS_FILE = f"{PROJECT_DIR}/execution_progress.json"

# Pipeline definitions - 36 total
PIPELINES = [
    # Phase 1: 2km (12 pipelines)
    {"name": "basic_2km_const", "phase": 1, "radius": "2km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_2km_rand", "phase": 1, "radius": "2km", "features": "basic", "attack": "rand_offset"},
    {"name": "basic_2km_withid_const", "phase": 1, "radius": "2km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_2km_withid_rand", "phase": 1, "radius": "2km", "features": "basic", "attack": "rand_offset"},
    {"name": "movement_2km_const", "phase": 1, "radius": "2km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_2km_rand", "phase": 1, "radius": "2km", "features": "movement", "attack": "rand_offset"},
    {"name": "movement_2km_withid_const", "phase": 1, "radius": "2km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_2km_withid_rand", "phase": 1, "radius": "2km", "features": "movement", "attack": "rand_offset"},
    {"name": "extended_2km_const", "phase": 1, "radius": "2km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_2km_rand", "phase": 1, "radius": "2km", "features": "extended", "attack": "rand_offset"},
    {"name": "extended_2km_withid_const", "phase": 1, "radius": "2km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_2km_withid_rand", "phase": 1, "radius": "2km", "features": "extended", "attack": "rand_offset"},
    
    # Phase 2: 100km (12 pipelines)
    {"name": "basic_100km_const", "phase": 2, "radius": "100km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_100km_rand", "phase": 2, "radius": "100km", "features": "basic", "attack": "rand_offset"},
    {"name": "basic_100km_withid_const", "phase": 2, "radius": "100km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_100km_withid_rand", "phase": 2, "radius": "100km", "features": "basic", "attack": "rand_offset"},
    {"name": "movement_100km_const", "phase": 2, "radius": "100km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_100km_rand", "phase": 2, "radius": "100km", "features": "movement", "attack": "rand_offset"},
    {"name": "movement_100km_withid_const", "phase": 2, "radius": "100km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_100km_withid_rand", "phase": 2, "radius": "100km", "features": "movement", "attack": "rand_offset"},
    {"name": "extended_100km_const", "phase": 2, "radius": "100km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_100km_rand", "phase": 2, "radius": "100km", "features": "extended", "attack": "rand_offset"},
    {"name": "extended_100km_withid_const", "phase": 2, "radius": "100km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_100km_withid_rand", "phase": 2, "radius": "100km", "features": "extended", "attack": "rand_offset"},
    
    # Phase 3: 200km (12 pipelines)
    {"name": "basic_200km_const", "phase": 3, "radius": "200km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_200km_rand", "phase": 3, "radius": "200km", "features": "basic", "attack": "rand_offset"},
    {"name": "basic_200km_withid_const", "phase": 3, "radius": "200km", "features": "basic", "attack": "const_offset_per_id"},
    {"name": "basic_200km_withid_rand", "phase": 3, "radius": "200km", "features": "basic", "attack": "rand_offset"},
    {"name": "movement_200km_const", "phase": 3, "radius": "200km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_200km_rand", "phase": 3, "radius": "200km", "features": "movement", "attack": "rand_offset"},
    {"name": "movement_200km_withid_const", "phase": 3, "radius": "200km", "features": "movement", "attack": "const_offset_per_id"},
    {"name": "movement_200km_withid_rand", "phase": 3, "radius": "200km", "features": "movement", "attack": "rand_offset"},
    {"name": "extended_200km_const", "phase": 3, "radius": "200km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_200km_rand", "phase": 3, "radius": "200km", "features": "extended", "attack": "rand_offset"},
    {"name": "extended_200km_withid_const", "phase": 3, "radius": "200km", "features": "extended", "attack": "const_offset_per_id"},
    {"name": "extended_200km_withid_rand", "phase": 3, "radius": "200km", "features": "extended", "attack": "rand_offset"},
]

def load_progress():
    """Load progress from file or start fresh"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "started_at": datetime.now().isoformat(),
        "current_pipeline": 0,
        "completed": [],
        "failed": [],
        "results": {}
    }

def save_progress(progress):
    """Save progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def run_preflight():
    """Run pre-flight checks"""
    print("=" * 60)
    print("PHASE 0: PRE-FLIGHT CHECKS")
    print("=" * 60)
    
    checks = []
    
    # Check 1: Config directory
    config_count = len(list(Path(CONFIG_DIR).glob("*.json")))
    checks.append(("Configs in production_configs_v2/", config_count == 36, config_count))
    
    # Check 2: Log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    checks.append(("Log directory exists", os.path.isdir(LOG_DIR), LOG_DIR))
    
    # Check 3: Results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checks.append(("Results directory exists", os.path.isdir(RESULTS_DIR), RESULTS_DIR))
    
    # Check 4: Memory check
    try:
        result = subprocess.run(['free', '-g'], capture_output=True, text=True)
        mem_line = [l for l in result.stdout.split('\n') if 'Mem:' in l][0]
        available = int(mem_line.split()[6])
        checks.append(("Available memory > 40GB", available > 40, f"{available}GB"))
    except:
        checks.append(("Memory check", False, "Could not check"))
    
    # Check 5: Disk space
    try:
        result = subprocess.run(['df', '-BG', PROJECT_DIR], capture_output=True, text=True)
        disk_line = result.stdout.split('\n')[1]
        available = int(disk_line.split()[3].replace('G', ''))
        checks.append(("Available disk > 200GB", available > 200, f"{available}GB"))
    except:
        checks.append(("Disk check", False, "Could not check"))
    
    # Print results
    all_passed = True
    for name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {value}")
        if not passed:
            all_passed = False
    
    return all_passed

def audit_log(log_file):
    """Audit a log file and extract key metrics"""
    audit = {
        "errors": [],
        "rows": None,
        "accuracy": None,
        "attackers_applied": False,
        "completed": False
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Check for errors
        for line in content.split('\n'):
            line_lower = line.lower()
            if any(err in line_lower for err in ['error', 'exception', 'traceback', 'failed']):
                if 'keyerror' in line_lower or 'valueerror' in line_lower or 'traceback' in line_lower:
                    audit["errors"].append(line.strip()[:200])
        
        # Check for row counts
        import re
        row_match = re.search(r'(\d+)\s*rows', content, re.IGNORECASE)
        if row_match:
            audit["rows"] = int(row_match.group(1))
        
        # Check for accuracy
        acc_match = re.search(r'accuracy[:\s]+(\d+\.?\d*)', content, re.IGNORECASE)
        if acc_match:
            audit["accuracy"] = float(acc_match.group(1))
        
        # Check for attackers
        if 'attacker' in content.lower() and ('applied' in content.lower() or 'malicious' in content.lower()):
            audit["attackers_applied"] = True
        
        # Check for completion
        if 'complete' in content.lower() or 'finished' in content.lower():
            audit["completed"] = True
            
    except Exception as e:
        audit["errors"].append(f"Failed to read log: {e}")
    
    return audit

def run_pipeline(pipeline, attempt=1, max_attempts=3):
    """Run a single pipeline with retry logic"""
    name = pipeline["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/{name}_{timestamp}.log"
    
    # Construct the run script path
    # Convert name to script: basic_2km_const -> Run2kmBasicConst.py
    parts = name.split('_')
    if 'withid' in name:
        # basic_2km_withid_const -> Run2kmBasicWithIdConst
        script_name = f"Run{parts[1].capitalize()}{parts[0].capitalize()}WithId{parts[3].capitalize()}.py"
    else:
        # basic_2km_const -> Run2kmBasicConst
        script_name = f"Run{parts[1].capitalize()}{parts[0].capitalize()}{parts[2].capitalize()}.py"
    
    script_path = f"{PROJECT_DIR}/{script_name}"
    
    print(f"\n{'='*60}")
    print(f"PIPELINE: {name} (Attempt {attempt}/{max_attempts})")
    print(f"Script: {script_name}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False, {"error": f"Script not found: {script_path}"}
    
    start_time = time.time()
    
    try:
        # Run the pipeline
        with open(log_file, 'w') as log:
            process = subprocess.run(
                ['python', script_path],
                cwd=PROJECT_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600  # 1 hour timeout
            )
        
        duration = time.time() - start_time
        
        # Audit the log
        audit = audit_log(log_file)
        audit["duration_seconds"] = int(duration)
        audit["log_file"] = log_file
        
        # Determine success
        if process.returncode == 0 and audit["completed"] and not audit["errors"]:
            print(f"✅ SUCCESS in {duration:.1f}s")
            if audit["accuracy"]:
                print(f"   Accuracy: {audit['accuracy']}")
            if audit["rows"]:
                print(f"   Rows: {audit['rows']}")
            return True, audit
        else:
            print(f"❌ FAILED (return code: {process.returncode})")
            if audit["errors"]:
                for err in audit["errors"][:3]:
                    print(f"   Error: {err}")
            
            # Retry?
            if attempt < max_attempts:
                print(f"   Retrying in 10 seconds...")
                time.sleep(10)
                return run_pipeline(pipeline, attempt + 1, max_attempts)
            
            return False, audit
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after 1 hour")
        return False, {"error": "Timeout after 1 hour"}
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        if attempt < max_attempts:
            print(f"   Retrying in 10 seconds...")
            time.sleep(10)
            return run_pipeline(pipeline, attempt + 1, max_attempts)
        return False, {"error": str(e)}

def main():
    print("=" * 60)
    print("AUTONOMOUS PIPELINE EXECUTOR")
    print(f"36 Pipelines | Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Load progress
    progress = load_progress()
    start_idx = progress["current_pipeline"]
    
    if start_idx > 0:
        print(f"\nResuming from pipeline {start_idx + 1}/36")
        print(f"Already completed: {len(progress['completed'])}")
        print(f"Already failed: {len(progress['failed'])}")
    
    # Run pre-flight (if starting fresh)
    if start_idx == 0:
        if not run_preflight():
            print("\n❌ PRE-FLIGHT FAILED - Aborting")
            sys.exit(1)
        print("\n✅ Pre-flight checks passed!")
    
    # Execute pipelines
    current_phase = 0
    for idx, pipeline in enumerate(PIPELINES[start_idx:], start=start_idx + 1):
        # Phase transition message
        if pipeline["phase"] != current_phase:
            current_phase = pipeline["phase"]
            print(f"\n{'#'*60}")
            print(f"# PHASE {current_phase}: {pipeline['radius'].upper()} PIPELINES")
            print(f"{'#'*60}")
        
        print(f"\n[{idx}/36] Starting {pipeline['name']}...")
        
        success, result = run_pipeline(pipeline)
        
        if success:
            progress["completed"].append(pipeline["name"])
        else:
            progress["failed"].append(pipeline["name"])
        
        progress["results"][pipeline["name"]] = result
        progress["current_pipeline"] = idx
        save_progress(progress)
        
        # Brief pause between pipelines
        time.sleep(5)
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Completed: {len(progress['completed'])}/36")
    print(f"Failed: {len(progress['failed'])}/36")
    
    if progress["failed"]:
        print("\nFailed pipelines:")
        for name in progress["failed"]:
            print(f"  ❌ {name}")
    
    progress["completed_at"] = datetime.now().isoformat()
    save_progress(progress)
    
    print(f"\nProgress saved to: {PROGRESS_FILE}")
    
    return len(progress["failed"]) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
