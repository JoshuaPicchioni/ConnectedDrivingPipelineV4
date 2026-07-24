#!/usr/bin/env python3
"""
Test the snapshot approach with singleton GeneratorContextProvider

This tests the enhanced cache key generation that captures context snapshots
instead of relying on provider instances, working around the singleton pattern.
"""

import sys
sys.path.append('/home/ubuntu/repos/ConnectedDrivingPipelineV4')

from Decorators.FileCache import create_deterministic_cache_key
from ServiceProviders.GeneratorContextProvider import GeneratorContextProvider

def test_snapshot_approach():
    """Test cache key generation with context snapshots."""
    
    print("🔍 Testing Snapshot Approach with Singleton GeneratorContextProvider...")
    print("=" * 70)
    
    # Test 1: Manually create context snapshots for different configurations
    print("\n1. Testing with context snapshots (dictionary approach):")
    
    # Simulate different configurations as context snapshots
    config1_snapshot = {
        'ConnectedDrivingAttacker.attack_ratio': 0.1,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID'
    }
    
    config2_snapshot = {
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID'
    }
    
    print(f"   Config 1 snapshot: {config1_snapshot}")
    print(f"   Config 2 snapshot: {config2_snapshot}")
    
    key1 = create_deterministic_cache_key('clean_data', ['test'], config1_snapshot)
    key2 = create_deterministic_cache_key('clean_data', ['test'], config2_snapshot)
    
    print(f"   Key 1 (10% attack): {key1}")
    print(f"   Key 2 (30% attack): {key2}")
    print(f"   ✅ Keys are unique: {key1 != key2}")
    
    # Test 2: Different spatial radii
    print("\n2. Testing different spatial radii:")
    
    config3_snapshot = {
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,  # 2km
        'attack_type': 'ConstOffsetPerID'
    }
    
    config4_snapshot = {
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 200000,  # 200km
        'attack_type': 'ConstOffsetPerID'
    }
    
    key3 = create_deterministic_cache_key('clean_data', ['test'], config3_snapshot)
    key4 = create_deterministic_cache_key('clean_data', ['test'], config4_snapshot)
    
    print(f"   Config 3 (2km):   {key3}")
    print(f"   Config 4 (200km): {key4}")
    print(f"   ✅ Keys are unique: {key3 != key4}")
    
    # Test 3: Test real-world scenario - simulate what would happen in pipeline
    print("\n3. Simulating real-world pipeline usage:")
    
    # Get the singleton provider (this is what the actual pipeline uses)
    provider = GeneratorContextProvider()
    
    # Simulate first pipeline run configuration
    provider.set({
        'ConnectedDrivingAttacker.attack_ratio': 0.1,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID',
        'dataset': 'wyoming_april_2021'
    })
    
    # Capture snapshot for first run
    snapshot1 = provider.getAll().copy()
    print(f"   Pipeline run 1 snapshot: {snapshot1}")
    
    # Simulate changing configuration for second run
    provider.set({
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID',
        'dataset': 'wyoming_april_2021'
    })
    
    # Capture snapshot for second run
    snapshot2 = provider.getAll().copy()
    print(f"   Pipeline run 2 snapshot: {snapshot2}")
    
    # Generate cache keys from snapshots
    pipeline_key1 = create_deterministic_cache_key('clean_data_with_timestamps', 
                                                  ['ConnectedDrivingCleaner', True, 'wyoming'], 
                                                  snapshot1)
    pipeline_key2 = create_deterministic_cache_key('clean_data_with_timestamps', 
                                                  ['ConnectedDrivingCleaner', True, 'wyoming'], 
                                                  snapshot2)
    
    print(f"   Pipeline key 1: {pipeline_key1}")
    print(f"   Pipeline key 2: {pipeline_key2}")
    print(f"   ✅ Pipeline keys are unique: {pipeline_key1 != pipeline_key2}")
    
    # Test 4: Same configuration should produce same key
    print("\n4. Testing reproducibility:")
    
    same_config = {
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID'
    }
    
    key5 = create_deterministic_cache_key('clean_data', ['test'], same_config)
    key6 = create_deterministic_cache_key('clean_data', ['test'], same_config)
    
    print(f"   Same config key 1: {key5}")
    print(f"   Same config key 2: {key6}")
    print(f"   ✅ Keys are identical: {key5 == key6}")
    
    print("\n" + "=" * 70)
    
    # Summary
    all_unique_keys = {key1, key2, key3, key4, pipeline_key1, pipeline_key2}
    expected_unique = 6  # All should be different except key5==key6
    
    print(f"🎯 SUMMARY:")
    print(f"   Unique keys generated: {len(all_unique_keys)}")
    print(f"   Expected unique: {expected_unique}")
    
    if len(all_unique_keys) == expected_unique:
        print(f"   🎉 SUCCESS: Snapshot approach eliminates cache collisions!")
        print(f"      Different configurations now produce unique cache keys.")
        print(f"      The singleton GeneratorContextProvider issue has been resolved.")
    else:
        print(f"   ⚠️  PARTIAL SUCCESS: Some improvements but issues remain.")

if __name__ == "__main__":
    test_snapshot_approach()