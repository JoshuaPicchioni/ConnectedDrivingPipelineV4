#!/usr/bin/env python3
"""
Task 5: Overfitting mitigation experiments.
constoffset, 100km, basic features, original center.
RF & DT with hyperparameter grid + LogisticRegression & SVM baselines.
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

PIPELINE_RESULTS_DIR = "/var/www/static/pipeline-results"
CONFIG_PATH = "production_configs_v2/basic_100km_const_pipeline_config.json"


def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    return {
        "train_f1": float(f1_score(y_train, train_pred, average='weighted')),
        "test_f1": float(f1_score(y_test, test_pred, average='weighted')),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_precision": float(precision_score(y_train, train_pred, average='weighted')),
        "test_precision": float(precision_score(y_test, test_pred, average='weighted')),
        "train_recall": float(recall_score(y_train, train_pred, average='weighted')),
        "test_recall": float(recall_score(y_test, test_pred, average='weighted')),
        "overfit_gap": float(f1_score(y_train, train_pred, average='weighted') - f1_score(y_test, test_pred, average='weighted')),
    }


def main():
    print(f"=== Overfitting Mitigation Experiments ===")
    print(f"Start: {datetime.now()}")
    
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    # Load cached data
    cache_path = config["cache"]["attack_dataset"]
    if not os.path.exists(cache_path):
        print(f"ERROR: Cache not found at {cache_path}. Run base pipeline first.")
        sys.exit(1)
    
    data_pd = pd.read_parquet(cache_path)
    features = config["ml"]["features"]
    label = config["ml"]["label"]
    
    X = data_pd[features].values
    y = data_pd[label].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Scale data for LR and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    all_results = []
    
    # RF hyperparameter grid
    max_depths = [5, 10, 20, None]
    min_samples_leafs = [1, 5, 10]
    
    print("\n--- Random Forest Grid ---")
    for md, msl in product(max_depths, min_samples_leafs):
        name = f"RF_depth{md}_leaf{msl}"
        print(f"  {name}...", end=" ", flush=True)
        clf = RandomForestClassifier(n_estimators=100, max_depth=md, min_samples_leaf=msl, random_state=42, n_jobs=-1)
        metrics = evaluate(clf, X_train, y_train, X_test, y_test)
        metrics["classifier"] = "RandomForest"
        metrics["max_depth"] = md
        metrics["min_samples_leaf"] = msl
        metrics["name"] = name
        all_results.append(metrics)
        print(f"Train F1={metrics['train_f1']:.4f}, Test F1={metrics['test_f1']:.4f}, Gap={metrics['overfit_gap']:.4f}")
    
    # DT hyperparameter grid
    print("\n--- Decision Tree Grid ---")
    for md, msl in product(max_depths, min_samples_leafs):
        name = f"DT_depth{md}_leaf{msl}"
        print(f"  {name}...", end=" ", flush=True)
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=42)
        metrics = evaluate(clf, X_train, y_train, X_test, y_test)
        metrics["classifier"] = "DecisionTree"
        metrics["max_depth"] = md
        metrics["min_samples_leaf"] = msl
        metrics["name"] = name
        all_results.append(metrics)
        print(f"Train F1={metrics['train_f1']:.4f}, Test F1={metrics['test_f1']:.4f}, Gap={metrics['overfit_gap']:.4f}")
    
    # Logistic Regression baseline
    print("\n--- Logistic Regression ---")
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    metrics = evaluate(clf, X_train_scaled, y_train, X_test_scaled, y_test)
    metrics["classifier"] = "LogisticRegression"
    metrics["max_depth"] = None
    metrics["min_samples_leaf"] = None
    metrics["name"] = "LogisticRegression"
    all_results.append(metrics)
    print(f"  Train F1={metrics['train_f1']:.4f}, Test F1={metrics['test_f1']:.4f}, Gap={metrics['overfit_gap']:.4f}")
    
    # SVM RBF baseline
    print("\n--- SVM (RBF kernel) ---")
    # Subsample if data is too large for SVM
    if len(X_train_scaled) > 50000:
        print(f"  Subsampling from {len(X_train_scaled)} to 50000 for SVM...")
        idx = np.random.RandomState(42).choice(len(X_train_scaled), 50000, replace=False)
        X_train_svm = X_train_scaled[idx]
        y_train_svm = y_train[idx]
    else:
        X_train_svm = X_train_scaled
        y_train_svm = y_train
    
    clf = SVC(kernel='rbf', random_state=42)
    metrics = evaluate(clf, X_train_svm, y_train_svm, X_test_scaled, y_test)
    metrics["classifier"] = "SVM_RBF"
    metrics["max_depth"] = None
    metrics["min_samples_leaf"] = None
    metrics["name"] = "SVM_RBF"
    all_results.append(metrics)
    print(f"  Train F1={metrics['train_f1']:.4f}, Test F1={metrics['test_f1']:.4f}, Gap={metrics['overfit_gap']:.4f}")
    
    # Save results
    results_dir = os.path.join(PIPELINE_RESULTS_DIR, "overfitting_basic_100km_constoffset_original")
    os.makedirs(results_dir, exist_ok=True)
    
    output = {
        "experiment": "overfitting_mitigation",
        "config": "basic_100km_const (original center)",
        "timestamp": datetime.now().isoformat(),
        "data_shape": {"train": list(X_train.shape), "test": list(X_test.shape)},
        "results": all_results,
    }
    
    with open(os.path.join(results_dir, "overfitting_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    # CSV
    with open(os.path.join(results_dir, "overfitting_results.csv"), "w") as f:
        f.write("name,classifier,max_depth,min_samples_leaf,train_f1,test_f1,overfit_gap,train_accuracy,test_accuracy\n")
        for r in all_results:
            f.write(f"{r['name']},{r['classifier']},{r['max_depth']},{r['min_samples_leaf']},{r['train_f1']:.4f},{r['test_f1']:.4f},{r['overfit_gap']:.4f},{r['train_accuracy']:.4f},{r['test_accuracy']:.4f}\n")
    
    # Summary
    with open(os.path.join(results_dir, "overfitting_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
