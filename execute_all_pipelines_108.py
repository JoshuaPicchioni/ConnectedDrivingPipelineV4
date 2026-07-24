#!/usr/bin/env python3
"""
Complete Pipeline Executor for ConnectedDrivingPipelineV4
Executes ALL 108 pipeline permutations

Created: 2026-02-24
Full matrix: 3 features × 3 radii × 6 attacks × 2 (with/without ID) = 108

Attack Types:
1. rand_offset - Random direction/distance per row
2. const_offset - Same direction/distance for ALL attackers  
3. const_offset_per_id - Random but consistent per vehicle ID
4. swap_rand - Swap positions randomly
5. override_const - Override to constant location
6. override_rand - Override to random location

Feature Sets:
1. basic - x_pos, y_pos, elevation (3 features)
2. movement - + speed, heading, accelYaw (6 features)
3. extended - + accuracy_semiMajor (7 features)
"""

import subprocess
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_DIR = "/home/ubuntu/repos/ConnectedDrivingPipelineV4"
LOG_DIR = f"{PROJECT_DIR}/logs"
PROGRESS_FILE = f"{PROJECT_DIR}/execution_progress_108.json"

# Attack types
ATTACK_TYPES = [
    "rand_offset",
    "const_offset", 
    "const_offset_per_id",
    "swap_rand",
    "override_const",
    "override_rand"
]

# Feature sets
FEATURE_SETS = ["basic", "movement", "extended"]

# Radii
RADII = ["2km", "100km", "200km"]

# Generate all 108 pipeline definitions
def generate_pipelines():
    pipelines = []
    phase = 1
    
    for radius in RADII:
        for features in FEATURE_SETS:
            for attack in ATTACK_TYPES:
                # Without vehicle ID as feature
                pipelines.append({
                    "name": f"{features}_{radius}_{attack.replace('_', '')}",
                    "phase": phase,
                    "radius": radius,
                    "features": features,
                    "attack": attack,
                    "with_id": False
                })
                
                # With vehicle ID as feature
                pipelines.append({
                    "name": f"{features}_{radius}_withid_{attack.replace('_', '')}",
                    "phase": phase,
                    "radius": radius,
                    "features": features,
                    "attack": attack,
                    "with_id": True
                })
        phase += 1
    
    return pipelines

PIPELINES = generate_pipelines()

print(f"Generated {len(PIPELINES)} pipeline configurations")
for i, p in enumerate(PIPELINES[:10]):
    print(f"  {i+1}. {p['name']} (radius={p['radius']}, features={p['features']}, attack={p['attack']}, with_id={p['with_id']})")
print(f"  ... and {len(PIPELINES) - 10} more")
