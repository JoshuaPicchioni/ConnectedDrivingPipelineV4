#!/usr/bin/env python3
"""
Auto-generated pipeline runner for extended_100km_withid_rand
Generated: 2026-02-23
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MachineLearning.DaskPipelineRunner import DaskPipelineRunner

def main():
    config_path = "production_configs_v2/extended_100km_withid_rand_pipeline_config.json"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Running pipeline: {config['pipeline_name']}")
    print(f"Config: {config_path}")
    
    runner = DaskPipelineRunner(config)
    runner.run()
    
    print(f"Pipeline {config['pipeline_name']} complete!")

if __name__ == "__main__":
    main()
