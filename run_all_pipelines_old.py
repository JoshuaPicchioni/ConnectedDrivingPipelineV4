#!/usr/bin/env python3
"""
Run All Pipelines - Sequential Execution of All 36 Pipeline Configurations

This script runs all pipeline configurations from production_configs_v2/
sequentially, collecting results and monitoring for errors.

Usage:
    python run_all_pipelines.py [--start-from N] [--only PATTERN]
    
Arguments:
    --start-from N: Start from pipeline N (1-indexed)
    --only PATTERN: Only run pipelines matching pattern (e.g., '2km', 'basic')
"""

import argparse
import json
import os
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


def get_all_configs():
    """Get all pipeline config files sorted alphabetically."""
    config_dir = 'production_configs_v2'
    configs = []
    
    for f in sorted(os.listdir(config_dir)):
        if f.endswith('_pipeline_config.json'):
            configs.append(os.path.join(config_dir, f))
    
    return configs


def run_pipeline(config_path: str, results_dir: str) -> dict:
    """Run a single pipeline and return results."""
    start_time = time.time()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pipeline_name = config.get('pipeline_name', 'unknown')
        print(f'Running: {pipeline_name}')
        print(f'  Config: {config_path}')
        
        runner = DaskPipelineRunner(config)
        results = runner.run()
        
        elapsed = time.time() - start_time
        
        # Save results
        result_data = {
            'pipeline_name': pipeline_name,
            'config_path': config_path,
            'status': 'success',
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
            'classifiers': []
        }
        
        for classifier, train_res, test_res in results:
            result_data['classifiers'].append({
                'name': classifier.__class__.__name__,
                'train_accuracy': train_res[0],
                'train_precision': train_res[1],
                'train_recall': train_res[2],
                'train_f1': train_res[3],
                'train_specificity': train_res[4],
                'test_accuracy': test_res[0],
                'test_precision': test_res[1],
                'test_recall': test_res[2],
                'test_f1': test_res[3],
                'test_specificity': test_res[4],
            })
        
        # Save to results file
        result_file = os.path.join(results_dir, f'{pipeline_name}_results.json')
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f'  SUCCESS in {elapsed:.1f}s')
        for r in result_data['classifiers']:
            print(f'    {r["name"]}: Train={r["train_accuracy"]:.4f}, Test={r["test_accuracy"]:.4f}')
        
        return result_data
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f'  FAILED: {str(e)}')
        
        return {
            'pipeline_name': os.path.basename(config_path).replace('_pipeline_config.json', ''),
            'config_path': config_path,
            'status': 'failed',
            'error': str(e),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description='Run all pipeline configurations')
    parser.add_argument('--start-from', type=int, default=1, help='Start from pipeline N (1-indexed)')
    parser.add_argument('--only', type=str, default=None, help='Only run pipelines matching pattern')
    parser.add_argument('--dry-run', action='store_true', help='List pipelines without running')
    args = parser.parse_args()
    
    # Get all configs
    configs = get_all_configs()
    
    # Filter if pattern specified
    if args.only:
        configs = [c for c in configs if args.only.lower() in c.lower()]
    
    # Apply start-from
    configs = configs[args.start_from - 1:]
    
    print('=' * 70)
    print('CONNECTED DRIVING PIPELINE RUNNER')
    print('=' * 70)
    print(f'Total pipelines: {len(configs)}')
    print(f'Start from: {args.start_from}')
    if args.only:
        print(f'Filter pattern: {args.only}')
    print()
    
    if args.dry_run:
        print('Pipelines to run:')
        for i, config in enumerate(configs, args.start_from):
            print(f'  {i}. {os.path.basename(config)}')
        return
    
    # Create results directory
    results_dir = 'pipeline-results'
    os.makedirs(results_dir, exist_ok=True)
    
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
            
            result = run_pipeline(config_path, results_dir)
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
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
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
    summary_file = os.path.join(results_dir, 'run_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'total': len(all_results),
            'successful': successful,
            'failed': failed,
            'total_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    print(f'\nResults saved to: {results_dir}/')


if __name__ == '__main__':
    main()
