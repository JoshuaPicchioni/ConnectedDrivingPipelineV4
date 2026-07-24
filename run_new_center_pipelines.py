#!/usr/bin/env python3
"""
Run new center point pipelines (Laramie and Evanston).
Runs one center at a time, checks storage, cleans caches between centers.

Usage:
    python run_new_center_pipelines.py [--center Laramie|Evanston|all] [--dry-run]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Add project root to path
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class TeeWriter:
    def __init__(self, capture, original):
        self.capture = capture
        self.original = original
    def write(self, msg):
        self.capture.write(msg)
        self.original.write(msg)
    def flush(self):
        self.capture.flush()
        self.original.flush()


@contextlib.contextmanager
def capture_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    sys.stdout = TeeWriter(stdout_capture, old_stdout)
    sys.stderr = TeeWriter(stderr_capture, old_stderr)
    try:
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


PIPELINE_RESULTS_DIR = "/var/www/static/pipeline-results"
CONFIG_DIR = "production_configs_v2/new_centers"
PROTECTED_FILES = [
    "April_2021_Wyoming_Data",
    "April_2021_Wyoming_Data_Fixed",
]


def check_storage(min_gb=50):
    """Check /home has at least min_gb free. Returns (free_gb, ok)."""
    stat = os.statvfs('/home')
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    return free_gb, free_gb >= min_gb


def clean_center_caches(center_name):
    """Remove cache directories for a specific center."""
    cache_base = "cache/matrix"
    cleaned = 0
    if os.path.exists(cache_base):
        for d in os.listdir(cache_base):
            if f"center{center_name}" in d:
                path = os.path.join(cache_base, d)
                print(f"  Cleaning cache: {path}")
                shutil.rmtree(path, ignore_errors=True)
                cleaned += 1
    print(f"  Cleaned {cleaned} cache directories for {center_name}")
    return cleaned


def run_single_pipeline(config_path, cluster_client):
    """Run a single pipeline and save results."""
    client = cluster_client
    
    with open(config_path) as f:
        config = json.load(f)
    
    pipeline_name = config["pipeline_name"]
    results_dir = os.path.join(PIPELINE_RESULTS_DIR, pipeline_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"Config: {config_path}")
    print(f"Start: {datetime.now()}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        with capture_output() as (stdout_cap, stderr_cap):
            # CRITICAL: Clear singletons before each pipeline to avoid state contamination
            from ClassTypes.SingletonABCMeta import SingletonABCMeta
            SingletonABCMeta._instances.clear()
            runner = DaskPipelineRunner.from_config(config_path)
            results = runner.run()
        
        elapsed = time.time() - start_time
        
        # Save log
        log_path = os.path.join(results_dir, "pipeline.log")
        with open(log_path, "w") as f:
            f.write(stdout_cap.getvalue())
            if stderr_cap.getvalue():
                f.write("\n\nSTDERR:\n")
                f.write(stderr_cap.getvalue())
        
        # Process results
        result_data = {
            "pipeline_name": pipeline_name,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "status": "success",
            "classifiers": {}
        }
        
        for classifier, train_result, test_result in results:
            # MDataClassifier wraps the actual classifier
            if hasattr(classifier, 'classifier'):
                clf_name = classifier.classifier.__class__.__name__
            else:
                clf_name = classifier.__class__.__name__
            result_data["classifiers"][clf_name] = {
                "train_accuracy": float(train_result[0]) if train_result else None,
                "train_precision": float(train_result[1]) if len(train_result) > 1 else None,
                "train_recall": float(train_result[2]) if len(train_result) > 2 else None,
                "train_f1": float(train_result[3]) if len(train_result) > 3 else None,
                "test_accuracy": float(test_result[0]) if test_result else None,
                "test_precision": float(test_result[1]) if len(test_result) > 1 else None,
                "test_recall": float(test_result[2]) if len(test_result) > 2 else None,
                "test_f1": float(test_result[3]) if len(test_result) > 3 else None,
            }
            print(f"  {clf_name}: Train F1={train_result[3]:.4f}, Test F1={test_result[3]:.4f}, Test Acc={test_result[0]:.4f}")
            
            # Confusion matrix
            try:
                if hasattr(classifier, 'test_predictions') and hasattr(classifier, 'test_labels'):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ConfusionMatrixDisplay.from_predictions(
                        classifier.test_labels, classifier.test_predictions, ax=ax
                    )
                    ax.set_title(f"{clf_name} - {pipeline_name}")
                    fig.savefig(os.path.join(results_dir, f"confusion_matrix_{clf_name}.png"), dpi=100, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not save confusion matrix for {clf_name}: {e}")
        
        # Save results JSON
        with open(os.path.join(results_dir, f"{pipeline_name}_results.json"), "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        
        # Save CSV
        csv_path = os.path.join(results_dir, f"{pipeline_name}.csv")
        with open(csv_path, "w") as f:
            f.write("pipeline_name,classifier,train_accuracy,train_precision,train_recall,train_f1,test_accuracy,test_precision,test_recall,test_f1,elapsed_seconds\n")
            for clf_name, metrics in result_data["classifiers"].items():
                f.write(f"{pipeline_name},{clf_name},{metrics['train_accuracy']},{metrics['train_precision']},{metrics['train_recall']},{metrics['train_f1']},{metrics['test_accuracy']},{metrics['test_precision']},{metrics['test_recall']},{metrics['test_f1']},{elapsed:.1f}\n")
        
        # Save summary
        with open(os.path.join(results_dir, "metrics_summary.txt"), "w") as f:
            f.write(f"Pipeline: {pipeline_name}\n")
            f.write(f"Completed: {datetime.now()}\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n\n")
            for clf_name, metrics in result_data["classifiers"].items():
                f.write(f"{clf_name}:\n")
                f.write(f"  Train F1: {metrics['train_f1']:.4f}\n")
                f.write(f"  Test F1:  {metrics['test_f1']:.4f}\n\n")
        
        print(f"  Completed in {elapsed:.1f}s")
        return True, pipeline_name, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_data = {
            "pipeline_name": pipeline_name,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        }
        with open(os.path.join(results_dir, f"{pipeline_name}_results.json"), "w") as f:
            json.dump(error_data, f, indent=2)
        
        return False, pipeline_name, elapsed


def run_center(center_name, configs, dry_run=False, skip_existing=False):
    """Run all pipelines for a single center."""
    print(f"\n{'#'*70}")
    print(f"# CENTER: {center_name} ({len(configs)} pipelines)")
    print(f"{'#'*70}")
    
    # Check storage
    free_gb, ok = check_storage(50)
    print(f"Storage: {free_gb:.1f}GB free")
    if not ok:
        print(f"ABORT: Less than 50GB free!")
        return []
    
    if dry_run:
        for c in configs:
            print(f"  [DRY RUN] Would run: {c}")
        return []
    
    results_summary = []
    
    for i, config_path in enumerate(configs):
        # Check if we should skip
        with open(config_path) as cf:
            pname = json.load(cf)["pipeline_name"]
        
        results_check_dir = os.path.join(PIPELINE_RESULTS_DIR, pname)
        results_check_json = os.path.join(results_check_dir, f"{pname}_results.json")
        if skip_existing and os.path.exists(results_check_json):
            try:
                with open(results_check_json) as rf:
                    rdata = json.load(rf)
                if rdata.get("status") == "success":
                    print(f"\n[{i+1}/{len(configs)}] SKIPPING {pname} (already has results)")
                    continue
            except:
                pass
        
        print(f"\n[{i+1}/{len(configs)}] Running {os.path.basename(config_path)}")
        
        # Storage check every 5 pipelines
        if i > 0 and i % 5 == 0:
            free_gb, ok = check_storage(50)
            print(f"  Storage check: {free_gb:.1f}GB free")
            if not ok:
                print(f"  ABORT: Less than 50GB free!")
                break
        
        # Fresh Dask cluster per pipeline to avoid state leakage
        print("  Starting fresh Dask cluster...")
        cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='12GB')
        client = Client(cluster)
        
        try:
            success, name, elapsed = run_single_pipeline(config_path, client)
            results_summary.append((success, name, elapsed))
        finally:
            client.close()
            cluster.close()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Summary
    succeeded = sum(1 for s, _, _ in results_summary if s)
    failed = sum(1 for s, _, _ in results_summary if not s)
    total_time = sum(e for _, _, e in results_summary)
    print(f"\n{center_name} Summary: {succeeded} succeeded, {failed} failed, {total_time:.0f}s total")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--center", default="all", choices=["Laramie", "Evanston", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip pipelines that already have results")
    args = parser.parse_args()
    
    # Collect configs by center
    all_configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*_pipeline_config.json")))
    
    laramie_configs = [c for c in all_configs if "centerLaramie" in c]
    evanston_configs = [c for c in all_configs if "centerEvanston" in c]
    
    print(f"Found {len(laramie_configs)} Laramie configs, {len(evanston_configs)} Evanston configs")
    
    all_results = {}
    
    if args.center in ("Laramie", "all"):
        all_results["Laramie"] = run_center("Laramie", laramie_configs, args.dry_run, args.skip_existing)
        
        if not args.dry_run:
            # Clean Laramie caches
            print("\nCleaning Laramie caches...")
            clean_center_caches("Laramie")
            free_gb, _ = check_storage(0)
            print(f"Storage after cleanup: {free_gb:.1f}GB free")
    
    if args.center in ("Evanston", "all"):
        all_results["Evanston"] = run_center("Evanston", evanston_configs, args.dry_run, args.skip_existing)
        
        if not args.dry_run:
            # Clean Evanston caches
            print("\nCleaning Evanston caches...")
            clean_center_caches("Evanston")
            free_gb, _ = check_storage(0)
            print(f"Storage after cleanup: {free_gb:.1f}GB free")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for center, results in all_results.items():
        if results:
            succeeded = sum(1 for s, _, _ in results if s)
            failed = sum(1 for s, _, _ in results if not s)
            print(f"{center}: {succeeded}/{len(results)} succeeded ({failed} failed)")
    
    print(f"\nDone: {datetime.now()}")


if __name__ == "__main__":
    main()
