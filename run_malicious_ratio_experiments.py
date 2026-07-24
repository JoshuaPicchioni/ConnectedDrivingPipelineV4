#!/usr/bin/env python3
"""
Task 6: Malicious ratio experiments.
constoffset, 100km, basic features, original center.
malicious_ratio = [0.05, 0.10, 0.20, 0.30]
"""
import json
import os
import sys
import copy
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dask
dask.config.set({
    'distributed.worker.memory.target': 0.7,
    'distributed.worker.memory.spill': 0.8,
    'distributed.worker.memory.pause': 0.9,
    'distributed.worker.memory.terminate': 0.95
})

from dask.distributed import Client, LocalCluster
from MachineLearning.DaskPipelineRunner import DaskPipelineRunner

PIPELINE_RESULTS_DIR = "/var/www/static/pipeline-results"
CONFIG_PATH = "production_configs_v2/basic_100km_const_pipeline_config.json"
RATIOS = [0.05, 0.10, 0.20, 0.30]


def main():
    print(f"=== Malicious Ratio Experiments ===")
    print(f"Start: {datetime.now()}")
    
    with open(CONFIG_PATH) as f:
        base_config = json.load(f)
    
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='12GB')
    client = Client(cluster)
    
    all_results = []
    
    try:
        for ratio in RATIOS:
            print(f"\n{'='*60}")
            print(f"Malicious Ratio: {ratio}")
            print(f"{'='*60}")
            
            # Create modified config
            config = copy.deepcopy(base_config)
            config["pipeline_name"] = f"malratio_{ratio}_basic_100km_constoffset_original"
            config["attack"]["malicious_ratio"] = ratio
            config["cache"]["clean_dataset"] = f"cache/matrix/{config['pipeline_name']}/clean.parquet"
            config["cache"]["attack_dataset"] = f"cache/matrix/{config['pipeline_name']}/attack.parquet"
            config["output"]["results_dir"] = f"results/matrix/{config['pipeline_name']}/"
            
            # Write temp config
            temp_config_path = f"/tmp/malratio_{ratio}_config.json"
            with open(temp_config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            start_time = time.time()
            
            try:
                runner = DaskPipelineRunner.from_config(temp_config_path)
                results = runner.run()
                elapsed = time.time() - start_time
                
                for classifier, train_result, test_result in results:
                    clf_name = classifier.__class__.__name__
                    entry = {
                        "malicious_ratio": ratio,
                        "classifier": clf_name,
                        "train_accuracy": float(train_result[0]),
                        "train_precision": float(train_result[1]),
                        "train_recall": float(train_result[2]),
                        "train_f1": float(train_result[3]),
                        "test_accuracy": float(test_result[0]),
                        "test_precision": float(test_result[1]),
                        "test_recall": float(test_result[2]),
                        "test_f1": float(test_result[3]),
                        "elapsed": elapsed,
                    }
                    all_results.append(entry)
                    print(f"  {clf_name}: Test F1={test_result[3]:.4f}, Test Acc={test_result[0]:.4f}")
                
                # Save per-ratio results
                ratio_dir = os.path.join(PIPELINE_RESULTS_DIR, config["pipeline_name"])
                os.makedirs(ratio_dir, exist_ok=True)
                with open(os.path.join(ratio_dir, f"{config['pipeline_name']}_results.json"), "w") as f:
                    json.dump({
                        "pipeline_name": config["pipeline_name"],
                        "malicious_ratio": ratio,
                        "results": [r for r in all_results if r["malicious_ratio"] == ratio],
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            # Clean this ratio's cache
            import shutil
            cache_dir = f"cache/matrix/{config['pipeline_name']}"
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"  Cleaned cache: {cache_dir}")
    
    finally:
        client.close()
        cluster.close()
    
    # Save combined results
    results_dir = os.path.join(PIPELINE_RESULTS_DIR, "malicious_ratio_experiment")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "malicious_ratio_results.json"), "w") as f:
        json.dump({
            "experiment": "malicious_ratio",
            "config_base": "basic_100km_const (original center)",
            "ratios": RATIOS,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2)
    
    with open(os.path.join(results_dir, "malicious_ratio_results.csv"), "w") as f:
        f.write("malicious_ratio,classifier,train_f1,test_f1,train_accuracy,test_accuracy,train_precision,test_precision,train_recall,test_recall\n")
        for r in all_results:
            f.write(f"{r['malicious_ratio']},{r['classifier']},{r['train_f1']:.4f},{r['test_f1']:.4f},{r['train_accuracy']:.4f},{r['test_accuracy']:.4f},{r['train_precision']:.4f},{r['test_precision']:.4f},{r['train_recall']:.4f},{r['test_recall']:.4f}\n")
    
    print(f"\nCombined results saved to {results_dir}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
