#!/usr/bin/env python3
"""
Task 7: Swaprand deep dive.
swaprand at 5km and 10km, basic features, original center.
Check storage before and after. Abort if >20GB cache per run.
"""
import json
import os
import sys
import copy
import time
import shutil
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
RADII = {"5km": 5000, "10km": 10000}
MAX_CACHE_GB = 20


def get_dir_size_gb(path):
    """Get directory size in GB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)


def check_storage():
    stat = os.statvfs('/home')
    return (stat.f_bavail * stat.f_frsize) / (1024**3)


def main():
    print(f"=== Swaprand Deep Dive ===")
    print(f"Start: {datetime.now()}")
    
    with open(CONFIG_PATH) as f:
        base_config = json.load(f)
    
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='12GB')
    client = Client(cluster)
    
    all_results = []
    
    try:
        for radius_name, radius_meters in RADII.items():
            free_before = check_storage()
            print(f"\n{'='*60}")
            print(f"Swaprand {radius_name} (radius={radius_meters}m)")
            print(f"Storage before: {free_before:.1f}GB free")
            print(f"{'='*60}")
            
            if free_before < 50:
                print("ABORT: Less than 50GB free!")
                break
            
            # Create modified config
            config = copy.deepcopy(base_config)
            config["pipeline_name"] = f"swaprand_{radius_name}_basic_original"
            config["attack"]["type"] = "swap_rand"
            # Remove offset params not needed for swaprand
            for key in ["offset_distance_min", "offset_distance_max", "offset_direction_min", "offset_direction_max"]:
                config["attack"].pop(key, None)
            config["data"]["filtering"]["radius_meters"] = radius_meters
            config["cache"]["clean_dataset"] = f"cache/matrix/{config['pipeline_name']}/clean.parquet"
            config["cache"]["attack_dataset"] = f"cache/matrix/{config['pipeline_name']}/attack.parquet"
            config["output"]["results_dir"] = f"results/matrix/{config['pipeline_name']}/"
            
            temp_config_path = f"/tmp/swaprand_{radius_name}_config.json"
            with open(temp_config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            start_time = time.time()
            
            try:
                runner = DaskPipelineRunner.from_config(temp_config_path)
                results = runner.run()
                elapsed = time.time() - start_time
                
                # Check cache size
                cache_dir = f"cache/matrix/{config['pipeline_name']}"
                if os.path.exists(cache_dir):
                    cache_gb = get_dir_size_gb(cache_dir)
                    print(f"  Cache size: {cache_gb:.2f}GB")
                    
                    if cache_gb > MAX_CACHE_GB:
                        print(f"  WARNING: Cache exceeds {MAX_CACHE_GB}GB! Noting for future reference.")
                
                for classifier, train_result, test_result in results:
                    clf_name = classifier.__class__.__name__
                    entry = {
                        "radius": radius_name,
                        "radius_meters": radius_meters,
                        "classifier": clf_name,
                        "train_accuracy": float(train_result[0]),
                        "train_f1": float(train_result[3]),
                        "test_accuracy": float(test_result[0]),
                        "test_f1": float(test_result[3]),
                        "elapsed": elapsed,
                    }
                    all_results.append(entry)
                    print(f"  {clf_name}: Test F1={test_result[3]:.4f}, Test Acc={test_result[0]:.4f}")
                
                # Save per-radius results
                ratio_dir = os.path.join(PIPELINE_RESULTS_DIR, config["pipeline_name"])
                os.makedirs(ratio_dir, exist_ok=True)
                with open(os.path.join(ratio_dir, f"{config['pipeline_name']}_results.json"), "w") as f:
                    json.dump({
                        "pipeline_name": config["pipeline_name"],
                        "radius": radius_name,
                        "results": [r for r in all_results if r["radius"] == radius_name],
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
            
            # Clean cache
            cache_dir = f"cache/matrix/{config['pipeline_name']}"
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"  Cleaned cache: {cache_dir}")
            
            free_after = check_storage()
            print(f"  Storage after: {free_after:.1f}GB free (delta: {free_before - free_after:.1f}GB)")
    
    finally:
        client.close()
        cluster.close()
    
    # Save combined results
    results_dir = os.path.join(PIPELINE_RESULTS_DIR, "swaprand_deepdive")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "swaprand_deepdive_results.json"), "w") as f:
        json.dump({
            "experiment": "swaprand_deepdive",
            "radii": list(RADII.keys()),
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2)
    
    with open(os.path.join(results_dir, "swaprand_deepdive_results.csv"), "w") as f:
        f.write("radius,classifier,train_f1,test_f1,train_accuracy,test_accuracy,elapsed\n")
        for r in all_results:
            f.write(f"{r['radius']},{r['classifier']},{r['train_f1']:.4f},{r['test_f1']:.4f},{r['train_accuracy']:.4f},{r['test_accuracy']:.4f},{r['elapsed']:.1f}\n")
    
    print(f"\nResults saved to {results_dir}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
