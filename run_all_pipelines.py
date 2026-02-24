#!/usr/bin/env python3
"""
Run All Pipelines - Sequential Execution of All 36 Pipeline Configurations

ENHANCED VERSION with comprehensive logging including:
- Vehicle ID counts (total, attacker, clean)
- Row counts at each stage
- Full detailed results.json with all statistics

CRITICAL FIX (2026-02-24): Captures stdout/stderr directly to pipeline.log
instead of relying on the broken log consolidation system.

Each pipeline run creates an organized folder with all artifacts:
  pipeline-results/{pipeline_name}/
  ├── {pipeline_name}_results.json    # Full results with timing, config, metrics, vehicle stats
  ├── {pipeline_name}.csv             # CSV format results
  ├── pipeline.log                    # FULL verbose pipeline execution log
  ├── confusion_matrix_RandomForest.png
  ├── confusion_matrix_DecisionTree.png
  ├── confusion_matrix_KNeighbors.png
  └── metrics_summary.txt             # Human-readable summary with vehicle stats

Usage:
    python run_all_pipelines.py [--start-from N] [--only PATTERN]
"""

import argparse
import contextlib
import glob
import io
import json
import os
import shutil
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

# For confusion matrix plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class TeeWriter:
    """Write to both a capture buffer and the original stream."""
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
    """Context manager to capture stdout and stderr while still printing to console."""
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


def get_all_configs():
    """Get all pipeline config files sorted alphabetically."""
    config_dir = 'production_configs_v2'
    configs = []
    
    for f in sorted(os.listdir(config_dir)):
        if f.endswith('_pipeline_config.json'):
            configs.append(os.path.join(config_dir, f))
    
    return configs


def generate_confusion_matrix_pngs(results, output_dir: str):
    """Generate confusion matrix PNG files for each classifier."""
    labels = ["Regular", "Malicious"]
    
    name_map = {
        'RandomForestClassifier': 'RandomForest',
        'DecisionTreeClassifier': 'DecisionTree',
        'KNeighborsClassifier': 'KNeighbors',
    }
    
    for mclassifier, train_res, test_res in results:
        full_name = mclassifier.classifier.__class__.__name__
        short_name = name_map.get(full_name, full_name)
        
        try:
            cm = mclassifier.get_confusion_matrix()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(ax=ax, cmap='Blues', values_format='.3f')
            
            ax.set_title(f'Confusion Matrix - {full_name}\n(Normalized)')
            
            png_path = os.path.join(output_dir, f'confusion_matrix_{short_name}.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f'    Generated: confusion_matrix_{short_name}.png')
            
        except Exception as e:
            print(f'    Warning: Could not generate confusion matrix for {full_name}: {e}')


def generate_metrics_summary(result_data: dict, output_dir: str):
    """Generate human-readable metrics_summary.txt with COMPREHENSIVE stats."""
    
    info_path = os.path.join(output_dir, 'metrics_summary.txt')
    
    with open(info_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write(f"PIPELINE RESULTS: {result_data['pipeline_name']}\n")
        f.write('=' * 80 + '\n\n')
        
        f.write(f"Status: {result_data['status'].upper()}\n")
        f.write(f"Timestamp: {result_data['timestamp']}\n")
        f.write(f"Elapsed Time: {result_data['elapsed_seconds']:.1f} seconds\n\n")
        
        # Configuration
        f.write('-' * 80 + '\n')
        f.write('CONFIGURATION\n')
        f.write('-' * 80 + '\n')
        config = result_data.get('config', {})
        f.write(f"Spatial Radius: {config.get('spatial_radius')} meters\n")
        f.write(f"Feature Set: {config.get('feature_set')}\n")
        f.write(f"Attack Type: {config.get('attack_type')}\n")
        f.write(f"With Vehicle ID: {config.get('with_vehicle_id')}\n")
        f.write(f"Malicious Ratio: {config.get('malicious_ratio')}\n")
        f.write(f"Center: ({config.get('center_latitude')}, {config.get('center_longitude')})\n")
        f.write(f"Date Range: {config.get('date_range', {})}\n\n")
        
        # Row counts
        f.write('-' * 80 + '\n')
        f.write('DATA STATISTICS\n')
        f.write('-' * 80 + '\n')
        f.write(f"Total Rows (after filtering): {result_data.get('total_rows_after_cleaning', 0):,}\n")
        f.write(f"Train Samples: {result_data.get('train_sample_size', 0):,}\n")
        f.write(f"Test Samples: {result_data.get('test_sample_size', 0):,}\n")
        f.write(f"Total Unique Vehicle IDs: {result_data.get('total_unique_vehicle_ids', 'N/A')}\n\n")
        
        # Train vehicle stats
        train_stats = result_data.get('train_vehicle_stats', {})
        if train_stats:
            f.write('-' * 80 + '\n')
            f.write('TRAIN SET VEHICLE STATISTICS\n')
            f.write('-' * 80 + '\n')
            f.write(f"Total Rows: {train_stats.get('total_rows', 'N/A'):,}\n")
            f.write(f"Total Unique Vehicle IDs: {train_stats.get('total_unique_vehicle_ids', 'N/A')}\n")
            f.write(f"Attacker Vehicle IDs: {train_stats.get('attacker_vehicle_count', 'N/A')}\n")
            f.write(f"Clean Vehicle IDs: {train_stats.get('clean_vehicle_count', 'N/A')}\n")
            f.write(f"Attacker Rows: {train_stats.get('attacker_row_count', 'N/A'):,}\n")
            f.write(f"Clean Rows: {train_stats.get('clean_row_count', 'N/A'):,}\n")
            
            # List attacker IDs if not too many
            attacker_ids = train_stats.get('attacker_vehicle_ids', [])
            if attacker_ids and len(attacker_ids) <= 30:
                f.write(f"Attacker Vehicle ID List: {attacker_ids}\n")
            elif attacker_ids:
                f.write(f"Attacker Vehicle ID List (first 20): {attacker_ids[:20]}...\n")
            f.write('\n')
        
        # Test vehicle stats  
        test_stats = result_data.get('test_vehicle_stats', {})
        if test_stats:
            f.write('-' * 80 + '\n')
            f.write('TEST SET VEHICLE STATISTICS\n')
            f.write('-' * 80 + '\n')
            f.write(f"Total Rows: {test_stats.get('total_rows', 'N/A'):,}\n")
            f.write(f"Total Unique Vehicle IDs: {test_stats.get('total_unique_vehicle_ids', 'N/A')}\n")
            f.write(f"Attacker Vehicle IDs: {test_stats.get('attacker_vehicle_count', 'N/A')}\n")
            f.write(f"Clean Vehicle IDs: {test_stats.get('clean_vehicle_count', 'N/A')}\n")
            f.write(f"Attacker Rows: {test_stats.get('attacker_row_count', 'N/A'):,}\n")
            f.write(f"Clean Rows: {test_stats.get('clean_row_count', 'N/A'):,}\n")
            
            attacker_ids = test_stats.get('attacker_vehicle_ids', [])
            if attacker_ids and len(attacker_ids) <= 30:
                f.write(f"Attacker Vehicle ID List: {attacker_ids}\n")
            elif attacker_ids:
                f.write(f"Attacker Vehicle ID List (first 20): {attacker_ids[:20]}...\n")
            f.write('\n')
        
        # Classifier results
        f.write('-' * 80 + '\n')
        f.write('CLASSIFIER RESULTS\n')
        f.write('-' * 80 + '\n')
        
        for clf in result_data.get('classifiers', []):
            f.write(f"\n{clf['name']}:\n")
            f.write(f"  TRAIN: Acc={clf['train_accuracy']:.4f}, Prec={clf['train_precision']:.4f}, ")
            f.write(f"Rec={clf['train_recall']:.4f}, F1={clf['train_f1']:.4f}, Spec={clf['train_specificity']:.4f}\n")
            f.write(f"  TEST:  Acc={clf['test_accuracy']:.4f}, Prec={clf['test_precision']:.4f}, ")
            f.write(f"Rec={clf['test_recall']:.4f}, F1={clf['test_f1']:.4f}, Spec={clf['test_specificity']:.4f}\n")
            f.write(f"  TIMING: Train={clf['total_train_time']:.3f}s, Predict={clf.get('prediction_test_time', 'N/A')}s\n")
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('END OF REPORT\n')
        f.write('=' * 80 + '\n')


def consolidate_logs(pipeline_name: str, output_dir: str, captured_output: str):
    """
    Write captured stdout/stderr directly to pipeline.log.
    
    CRITICAL FIX: Instead of searching for log files in logs/{pipeline_name}/,
    we now capture stdout/stderr directly during pipeline execution.
    This bypasses the broken dependency injection in the Logger class.
    
    Also appends any additional log files found.
    """
    log_dst = os.path.join(output_dir, 'pipeline.log')
    
    log_content = []
    
    # First, add the captured stdout/stderr (this is the main content now)
    log_content.append('=' * 80)
    log_content.append(f'PIPELINE EXECUTION LOG: {pipeline_name}')
    log_content.append(f'Generated: {datetime.now().isoformat()}')
    log_content.append('=' * 80)
    log_content.append('')
    log_content.append(captured_output)
    
    # Also search for any additional log files
    # Search in multiple locations where logs might be
    search_patterns = [
        f'logs/{pipeline_name}/*.txt',
        f'logs/*/{pipeline_name}.txt',  # Logs might be in wrong directory
    ]
    
    found_files = set()
    for pattern in search_patterns:
        for log_path in glob.glob(pattern):
            if log_path not in found_files:
                found_files.add(log_path)
    
    if found_files:
        log_content.append('')
        log_content.append('=' * 80)
        log_content.append('ADDITIONAL LOG FILES')
        log_content.append('=' * 80)
        
        for log_path in sorted(found_files):
            log_content.append(f'\n--- {os.path.basename(log_path)} ---\n')
            try:
                with open(log_path, 'r') as infile:
                    log_content.append(infile.read())
            except Exception as e:
                log_content.append(f'Error reading {log_path}: {e}\n')
    
    # Write consolidated log
    with open(log_dst, 'w') as outfile:
        outfile.write('\n'.join(log_content))
    
    print(f'    Saved: pipeline.log ({len(captured_output)} chars captured)')


def run_pipeline(config_path: str, base_results_dir: str) -> dict:
    """Run a single pipeline and return comprehensive results."""
    start_time = time.time()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # CRITICAL FIX: Clear logs for this specific pipeline BEFORE running
        # This ensures each pipeline gets fresh logs that are not mixed with others
        pipeline_name = config.get('name', os.path.basename(config_path).replace('.json', ''))
        pipeline_log_dir = f"logs/{pipeline_name}"
        if os.path.exists(pipeline_log_dir):
            import shutil
            shutil.rmtree(pipeline_log_dir)
            print(f"    Cleared old logs: {pipeline_log_dir}")
        os.makedirs(pipeline_log_dir, exist_ok=True)
        
        pipeline_name = config.get('pipeline_name', 'unknown')
        print(f'Running: {pipeline_name}')
        print(f'  Config: {config_path}')
        
        # Create per-pipeline output directory
        pipeline_output_dir = os.path.join(base_results_dir, pipeline_name)
        os.makedirs(pipeline_output_dir, exist_ok=True)
        
        # CRITICAL FIX: Capture all output during pipeline execution
        with capture_output() as (stdout_cap, stderr_cap):
            runner = DaskPipelineRunner(config)
            results, metadata = runner.run_with_metadata()
        
        # Get captured output
        captured_stdout = stdout_cap.getvalue()
        captured_stderr = stderr_cap.getvalue()
        combined_output = captured_stdout
        if captured_stderr:
            combined_output += '\n\n=== STDERR ===\n' + captured_stderr
        
        elapsed = time.time() - start_time
        
        # Extract config info
        pipeline_config = config.get('pipeline', {})
        data_config = config.get('data', {})
        attack_config = config.get('attack', {})
        ml_config = config.get('ml', {})
        filtering = data_config.get('filtering', {})
        
        # Build comprehensive result data
        result_data = {
            'pipeline_name': pipeline_name,
            'config_path': config_path,
            'status': 'success',
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
            
            # Pipeline configuration
            'config': {
                'spatial_radius': filtering.get('radius_meters', pipeline_config.get('spatial_radius')),
                'feature_set': pipeline_config.get('feature_set'),
                'attack_type': attack_config.get('type'),
                'with_vehicle_id': pipeline_config.get('with_vehicle_id', False),
                'malicious_ratio': attack_config.get('malicious_ratio', 0.0),
                'center_latitude': filtering.get('center_latitude'),
                'center_longitude': filtering.get('center_longitude'),
                'date_range': data_config.get('date_range', {}),
            },
            
            # Sample sizes from metadata
            'train_sample_size': metadata.get('train_sample_size', 0),
            'test_sample_size': metadata.get('test_sample_size', 0),
            'total_rows_after_cleaning': metadata.get('total_rows', metadata.get('filtered_row_count', 0)),
            'total_unique_vehicle_ids': metadata.get('total_unique_vehicle_ids', 0),
            
            # ENHANCED: Vehicle statistics from metadata
            'train_vehicle_stats': metadata.get('train_vehicle_stats', {}),
            'test_vehicle_stats': metadata.get('test_vehicle_stats', {}),
            
            # Classifier results
            'classifiers': []
        }
        
        for mclassifier, train_res, test_res in results:
            classifier_name = mclassifier.classifier.__class__.__name__
            
            train_time = getattr(mclassifier, 'elapsed_train_time', -1)
            prediction_time = getattr(mclassifier, 'elapsed_prediction_time', -1)
            prediction_train_time = getattr(mclassifier, 'elapsed_prediction_train_time', -1)
            
            train_size = metadata.get('train_sample_size', 1)
            test_size = metadata.get('test_sample_size', 1)
            
            result_data['classifiers'].append({
                'name': classifier_name,
                
                # Timing metrics
                'total_train_time': train_time,
                'train_time_per_sample': train_time / train_size if train_size > 0 and train_time > 0 else 0,
                'prediction_test_time': prediction_time,
                'prediction_test_time_per_sample': prediction_time / test_size if test_size > 0 and prediction_time > 0 else 0,
                'prediction_train_time': prediction_train_time,
                'prediction_train_time_per_sample': prediction_train_time / train_size if train_size > 0 and prediction_train_time > 0 else 0,
                
                # Train metrics
                'train_accuracy': train_res[0],
                'train_precision': train_res[1],
                'train_recall': train_res[2],
                'train_f1': train_res[3],
                'train_specificity': train_res[4],
                
                # Test metrics
                'test_accuracy': test_res[0],
                'test_precision': test_res[1],
                'test_recall': test_res[2],
                'test_f1': test_res[3],
                'test_specificity': test_res[4],
            })
        
        # === SAVE ALL ARTIFACTS TO PIPELINE OUTPUT DIR ===
        
        # 1. Save JSON results (COMPREHENSIVE)
        result_file = os.path.join(pipeline_output_dir, f'{pipeline_name}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        print(f'    Saved: {pipeline_name}_results.json')
        
        # 2. Generate confusion matrix PNGs
        print(f'  Generating confusion matrices...')
        generate_confusion_matrix_pngs(results, pipeline_output_dir)
        
        # 3. Copy CSV if it exists
        csv_src = f'Outputs/Output/{pipeline_name}.csv'
        if os.path.exists(csv_src):
            csv_dst = os.path.join(pipeline_output_dir, f'{pipeline_name}.csv')
            shutil.copy2(csv_src, csv_dst)
            print(f'    Copied: {pipeline_name}.csv')
        
        # 4. CRITICAL FIX: Write captured output to pipeline.log
        consolidate_logs(pipeline_name, pipeline_output_dir, combined_output)
        
        # 5. Generate ENHANCED metrics_summary.txt
        generate_metrics_summary(result_data, pipeline_output_dir)
        print(f'    Generated: metrics_summary.txt')
        
        # Print summary
        print(f'  SUCCESS in {elapsed:.1f}s')
        print(f'    Output: {pipeline_output_dir}/')
        print(f'    Samples: {result_data["train_sample_size"]:,} train, {result_data["test_sample_size"]:,} test')
        
        # Print vehicle stats
        train_stats = result_data.get('train_vehicle_stats', {})
        test_stats = result_data.get('test_vehicle_stats', {})
        print(f'    Train vehicles: {train_stats.get("total_unique_vehicle_ids", "?")} total, {train_stats.get("attacker_vehicle_count", "?")} attackers')
        print(f'    Test vehicles: {test_stats.get("total_unique_vehicle_ids", "?")} total, {test_stats.get("attacker_vehicle_count", "?")} attackers')
        
        for r in result_data['classifiers']:
            print(f'    {r["name"]}: Train={r["train_accuracy"]:.4f}, Test={r["test_accuracy"]:.4f}, TrainTime={r["total_train_time"]:.3f}s')
        
        return result_data
        
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f'  FAILED: {error_msg}')
        print(tb)
        
        pipeline_name = os.path.basename(config_path).replace('_pipeline_config.json', '')
        
        # Still save error log
        pipeline_output_dir = os.path.join(base_results_dir, pipeline_name)
        os.makedirs(pipeline_output_dir, exist_ok=True)
        
        error_log = f"PIPELINE FAILED: {pipeline_name}\n"
        error_log += f"Error: {error_msg}\n\n"
        error_log += f"Traceback:\n{tb}\n"
        
        with open(os.path.join(pipeline_output_dir, 'pipeline.log'), 'w') as f:
            f.write(error_log)
        
        return {
            'pipeline_name': pipeline_name,
            'config_path': config_path,
            'status': 'failed',
            'error': error_msg,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run all pipeline configurations')
    parser.add_argument('--start-from', type=int, default=1, help='Start from pipeline N (1-indexed)')
    parser.add_argument('--only', type=str, default=None, help='Only run pipelines matching pattern')
    parser.add_argument('--dry-run', action='store_true', help='List pipelines without running')
    parser.add_argument('--results-dir', type=str, default='pipeline-results', help='Base results directory')
    args = parser.parse_args()
    
    # Get all configs
    configs = get_all_configs()
    
    # Filter if pattern specified
    if args.only:
        configs = [c for c in configs if args.only.lower() in c.lower()]
    
    # Apply start-from
    configs = configs[args.start_from - 1:]
    
    print('=' * 80)
    print('CONNECTED DRIVING PIPELINE RUNNER (ENHANCED)')
    print('=' * 80)
    print(f'Total pipelines: {len(configs)}')
    print(f'Start from: {args.start_from}')
    print(f'Results dir: {args.results_dir}')
    if args.only:
        print(f'Filter pattern: {args.only}')
    print()
    
    if args.dry_run:
        print('Pipelines to run:')
        for i, config in enumerate(configs, args.start_from):
            print(f'  {i}. {os.path.basename(config)}')
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Setup Dask cluster
    print('Setting up Dask cluster...')
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=3,
        memory_limit='12GB',
        dashboard_address=None
    )
    client = Client(cluster)
    print(f'Dask dashboard: {client.dashboard_link}')
    print()
    
    # Run all pipelines
    all_results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    try:
        for i, config_path in enumerate(configs, args.start_from):
            print()
            print(f'[{i}/{len(configs) + args.start_from - 1}] ', end='')
            
            result = run_pipeline(config_path, args.results_dir)
            all_results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
    
    finally:
        client.close()
        cluster.close()
    
    total_time = time.time() - start_time
    
    # Summary
    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total pipelines: {len(all_results)}')
    print(f'Successful: {successful}')
    print(f'Failed: {failed}')
    print(f'Total time: {total_time/60:.1f} minutes')
    
    if failed > 0:
        print()
        print('FAILED PIPELINES:')
        for r in all_results:
            if r['status'] == 'failed':
                print(f'  - {r["pipeline_name"]}: {r["error"]}')
    
    # Save summary
    summary_file = os.path.join(args.results_dir, 'run_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'total': len(all_results),
            'successful': successful,
            'failed': failed,
            'total_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f'\nResults saved to: {args.results_dir}/')
    print(f'Each pipeline has its own folder with all artifacts.')


if __name__ == '__main__':
    main()
