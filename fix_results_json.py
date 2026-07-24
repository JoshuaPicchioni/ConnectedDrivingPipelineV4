#!/usr/bin/env python3
"""
Post-process: Fix results JSON files by parsing the pipeline.log to get per-classifier metrics.
Handles the MDataClassifier wrapper naming issue.
"""
import json
import os
import re
import sys
from glob import glob

RESULTS_DIR = "/var/www/static/pipeline-results"

def parse_pipeline_log(log_path):
    """Parse pipeline.log to extract per-classifier metrics."""
    with open(log_path) as f:
        lines = f.readlines()
    
    classifiers = {}
    current_clf = None
    metrics = {}
    is_train = True  # First set of metrics after CLASSIFIER is train
    
    for line in lines:
        # Match CLASSIFIER: name
        m = re.search(r'CLASSIFIER:\s+(\w+)', line)
        if m:
            if current_clf and metrics:
                classifiers[current_clf] = dict(metrics)
            current_clf = m.group(1)
            metrics = {}
            is_train = True
            continue
        
        if current_clf:
            # Match metrics
            if 'Accuracy:' in line:
                val = float(re.search(r'Accuracy:\s+([\d.]+)', line).group(1))
                if is_train:
                    metrics['train_accuracy'] = val
                else:
                    metrics['test_accuracy'] = val
            elif 'Precision:' in line:
                val = float(re.search(r'Precision:\s+([\d.]+)', line).group(1))
                if is_train:
                    metrics['train_precision'] = val
                else:
                    metrics['test_precision'] = val
            elif 'Recall:' in line:
                val = float(re.search(r'Recall:\s+([\d.]+)', line).group(1))
                if is_train:
                    metrics['train_recall'] = val
                else:
                    metrics['test_recall'] = val
            elif 'F1 Score:' in line:
                val = float(re.search(r'F1 Score:\s+([\d.]+)', line).group(1))
                if is_train:
                    metrics['train_f1'] = val
                else:
                    metrics['test_f1'] = val
            elif 'Specificity:' in line:
                val = float(re.search(r'Specificity:\s+([\d.]+)', line).group(1))
                if is_train:
                    metrics['train_specificity'] = val
                else:
                    metrics['test_specificity'] = val
            
            # After we see test_accuracy, switch to next section
            if 'test_f1' in metrics and 'test_specificity' in metrics:
                # This classifier is complete, reset for potential next one
                pass
            elif 'train_specificity' in metrics and 'test_accuracy' not in metrics:
                is_train = False
    
    # Don't forget the last classifier
    if current_clf and metrics:
        classifiers[current_clf] = dict(metrics)
    
    return classifiers


def fix_results_dir(results_subdir):
    """Fix a single pipeline's results JSON."""
    results_json_files = glob(os.path.join(results_subdir, "*_results.json"))
    log_file = os.path.join(results_subdir, "pipeline.log")
    
    if not results_json_files or not os.path.exists(log_file):
        return False
    
    # Parse log
    classifiers = parse_pipeline_log(log_file)
    if not classifiers:
        return False
    
    # Update results JSON
    for rjf in results_json_files:
        with open(rjf) as f:
            data = json.load(f)
        
        if data.get("status") != "success":
            continue
        
        # Replace classifiers section
        data["classifiers"] = classifiers
        
        with open(rjf, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # Also fix CSV
    pipeline_name = os.path.basename(results_subdir)
    csv_path = os.path.join(results_subdir, f"{pipeline_name}.csv")
    elapsed = data.get("elapsed_seconds", 0)
    
    with open(csv_path, 'w') as f:
        f.write("pipeline_name,classifier,train_accuracy,train_precision,train_recall,train_f1,test_accuracy,test_precision,test_recall,test_f1,elapsed_seconds\n")
        for clf_name, metrics in classifiers.items():
            f.write(f"{pipeline_name},{clf_name},{metrics.get('train_accuracy','')},{metrics.get('train_precision','')},{metrics.get('train_recall','')},{metrics.get('train_f1','')},{metrics.get('test_accuracy','')},{metrics.get('test_precision','')},{metrics.get('test_recall','')},{metrics.get('test_f1','')},{elapsed:.1f}\n")
    
    return True


def main():
    fixed = 0
    for subdir in sorted(os.listdir(RESULTS_DIR)):
        full_path = os.path.join(RESULTS_DIR, subdir)
        if not os.path.isdir(full_path):
            continue
        if "center" not in subdir:
            continue  # Only fix new center results
        
        if fix_results_dir(full_path):
            print(f"Fixed: {subdir}")
            fixed += 1
    
    print(f"\nFixed {fixed} result directories")


if __name__ == "__main__":
    main()
