#!/usr/bin/env python3
"""
Task 4: Cross-validation experiment.
5-fold CV on basic features, 100km, constoffset, original center.
"""
import json
import os
import sys
import time
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PIPELINE_RESULTS_DIR = "/var/www/static/pipeline-results"
# Use the existing basic_100km_const config (original center)
CONFIG_PATH = "production_configs_v2/basic_100km_const_pipeline_config.json"


def main():
    print(f"=== Cross-Validation Experiment ===")
    print(f"Start: {datetime.now()}")
    
    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    print(f"Config: {config['pipeline_name']}")
    
    # Set up Dask
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='12GB')
    client = Client(cluster)
    
    try:
        # Load and prepare data using the pipeline
        runner = DaskPipelineRunner.from_config(CONFIG_PATH)
        
        # We need to get data up to the attack stage, then do CV manually
        # Run the full pipeline once to get the attacked data
        print("Running pipeline to get processed data...")
        results_with_meta = runner.run_with_metadata()
        results, metadata = results_with_meta
        
        # Now we need to re-load the cached attacked data for CV
        cache_config = config.get("cache", {})
        attack_cache = cache_config.get("attack_dataset")
        
        import pandas as pd
        
        if attack_cache and os.path.exists(attack_cache):
            print(f"Loading cached attack data from {attack_cache}")
            data_pd = pd.read_parquet(attack_cache)
        else:
            print("No cache found, using data from pipeline run")
            # Fallback: re-run just the data gathering part
            # This is less ideal but works
            print("ERROR: Need cached data for CV. Run the base pipeline first.")
            sys.exit(1)
        
        features = config["ml"]["features"]
        label = config["ml"]["label"]
        
        X = data_pd[features].values
        y = data_pd[label].values
        
        print(f"Data shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y.astype(int))}")
        
        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        classifiers = {
            "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
            "KNeighbors": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        }
        
        cv_results = {name: {"f1": [], "accuracy": [], "precision": [], "recall": []} for name in classifiers}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold+1}/5 ---")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            for name, clf_factory in classifiers.items():
                clf = clf_factory()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                f1 = f1_score(y_test, y_pred, average='weighted')
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                
                cv_results[name]["f1"].append(f1)
                cv_results[name]["accuracy"].append(acc)
                cv_results[name]["precision"].append(prec)
                cv_results[name]["recall"].append(rec)
                
                print(f"  {name}: F1={f1:.4f}, Acc={acc:.4f}")
        
        # Summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY (5-fold)")
        print(f"{'='*60}")
        
        summary = {}
        for name in classifiers:
            metrics = cv_results[name]
            summary[name] = {
                "f1_mean": float(np.mean(metrics["f1"])),
                "f1_std": float(np.std(metrics["f1"])),
                "accuracy_mean": float(np.mean(metrics["accuracy"])),
                "accuracy_std": float(np.std(metrics["accuracy"])),
                "precision_mean": float(np.mean(metrics["precision"])),
                "precision_std": float(np.std(metrics["precision"])),
                "recall_mean": float(np.mean(metrics["recall"])),
                "recall_std": float(np.std(metrics["recall"])),
                "per_fold_f1": [float(x) for x in metrics["f1"]],
            }
            print(f"{name}:")
            print(f"  F1:  {summary[name]['f1_mean']:.4f} ± {summary[name]['f1_std']:.4f}")
            print(f"  Acc: {summary[name]['accuracy_mean']:.4f} ± {summary[name]['accuracy_std']:.4f}")
        
        # Save results
        results_dir = os.path.join(PIPELINE_RESULTS_DIR, "cv_basic_100km_constoffset_original")
        os.makedirs(results_dir, exist_ok=True)
        
        output = {
            "experiment": "5-fold_cross_validation",
            "config": "basic_100km_const (original center)",
            "timestamp": datetime.now().isoformat(),
            "n_folds": 5,
            "data_shape": list(X.shape),
            "classifiers": summary,
        }
        
        with open(os.path.join(results_dir, "cv_results.json"), "w") as f:
            json.dump(output, f, indent=2)
        
        # CSV
        with open(os.path.join(results_dir, "cv_results.csv"), "w") as f:
            f.write("classifier,metric,mean,std,fold1,fold2,fold3,fold4,fold5\n")
            for name, s in summary.items():
                folds = ",".join(f"{x:.4f}" for x in s["per_fold_f1"])
                f.write(f"{name},f1,{s['f1_mean']:.4f},{s['f1_std']:.4f},{folds}\n")
                f.write(f"{name},accuracy,{s['accuracy_mean']:.4f},{s['accuracy_std']:.4f}\n")
        
        print(f"\nResults saved to {results_dir}")
    
    finally:
        client.close()
        cluster.close()
    
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
