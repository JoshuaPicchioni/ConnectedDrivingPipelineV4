#!/usr/bin/env python3
"""
Queue all 36 pipelines to the dashboard daemon
This ensures results go to /var/www/static/pipeline-results/
"""

import json
import os
from datetime import datetime

# Path to queue file
QUEUE_FILE = os.path.expanduser("~/.pipeline-queue/queue.json")

# All 36 pipeline scripts (in order)
PIPELINES = [
    # Phase 1: 2km (12 pipelines)
    "Run2kmBasicConst.py",
    "Run2kmBasicRand.py",
    "Run2kmBasicWithIdConst.py",
    "Run2kmBasicWithIdRand.py",
    "Run2kmMovementConst.py",
    "Run2kmMovementRand.py",
    "Run2kmMovementWithIdConst.py",
    "Run2kmMovementWithIdRand.py",
    "Run2kmExtendedConst.py",
    "Run2kmExtendedRand.py",
    "Run2kmExtendedWithIdConst.py",
    "Run2kmExtendedWithIdRand.py",
    
    # Phase 2: 100km (12 pipelines)
    "Run100kmBasicConst.py",
    "Run100kmBasicRand.py",
    "Run100kmBasicWithIdConst.py",
    "Run100kmBasicWithIdRand.py",
    "Run100kmMovementConst.py",
    "Run100kmMovementRand.py",
    "Run100kmMovementWithIdConst.py",
    "Run100kmMovementWithIdRand.py",
    "Run100kmExtendedConst.py",
    "Run100kmExtendedRand.py",
    "Run100kmExtendedWithIdConst.py",
    "Run100kmExtendedWithIdRand.py",
    
    # Phase 3: 200km (12 pipelines)
    "Run200kmBasicConst.py",
    "Run200kmBasicRand.py",
    "Run200kmBasicWithIdConst.py",
    "Run200kmBasicWithIdRand.py",
    "Run200kmMovementConst.py",
    "Run200kmMovementRand.py",
    "Run200kmMovementWithIdConst.py",
    "Run200kmMovementWithIdRand.py",
    "Run200kmExtendedConst.py",
    "Run200kmExtendedRand.py",
    "Run200kmExtendedWithIdConst.py",
    "Run200kmExtendedWithIdRand.py",
]

def main():
    today = datetime.now().strftime("%Y%m%d")
    
    # Create queue entries
    # NOTE: Pipeline names should be JUST the filename, not the full path
    # The daemon runs from REPO_DIR and expects scripts to be there
    queue = []
    for i, pipeline in enumerate(PIPELINES):
        job = {
            "id": f"{today}_v4_{i}",  # v4 for this run
            "pipeline": pipeline,  # Just the filename, e.g., "Run2kmBasicConst.py"
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        queue.append(job)
        print(f"Added: {job['id']} -> {pipeline}")
    
    # Write queue file
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)
    
    print(f"\n✅ Queued {len(queue)} pipelines to {QUEUE_FILE}")
    print("The dashboard daemon will process them automatically.")
    print("\nTo monitor progress:")
    print("  watch -n 5 'cat ~/.pipeline-queue/state.json | python3 -c \"import json,sys; d=json.load(sys.stdin); print(f\\\"Current: {d.get(chr(39)current_job chr(39))}\\nHistory: {len(d.get(chr(39)history chr(39),[]))}\\\")'")

if __name__ == "__main__":
    main()
