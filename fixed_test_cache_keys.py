#!/usr/bin/env python3
"""
Fixed Cache Key Validation Script

Tests the enhanced cache key generation to ensure different configurations
produce unique cache keys, preventing cache collision issues.
"""

import sys
import os

# Add the repo directory to path so we can import the modules
sys.path.append('/home/ubuntu/repos/ConnectedDrivingPipelineV4')

from Decorators.FileCache import create_deterministic_cache_key
from ServiceProviders.GeneratorContextProvider import GeneratorContextProvider

def test_cache_key_uniqueness():
    """Test that different configurations produce unique cache keys."""
    
    print("🔍 Testing Cache Key Uniqueness (FIXED VERSION)...")
    print("=" * 60)
    
    # Test 1: Different attack ratios
    print("\n1. Testing different attack ratios:")
    
    # Create separate GeneratorContextProvider instances
    config1 = GeneratorContextProvider()
    config1.add('ConnectedDrivingAttacker.attack_ratio', 0.1)
    config1.add('ConnectedDrivingLargeDataCleaner.max_dist', 2000)
    config1.add('attack_type', 'ConstOffsetPerID')
    
    config2 = GeneratorContextProvider()
    config2.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config2.add('ConnectedDrivingLargeDataCleaner.max_dist', 2000)
    config2.add('attack_type', 'ConstOffsetPerID')
    
    print(f"   Config 1 contexts: {config1.getAll()}")
    print(f"   Config 2 contexts: {config2.getAll()}")
    
    key1 = create_deterministic_cache_key('clean_data', ['test'], config1)
    key2 = create_deterministic_cache_key('clean_data', ['test'], config2)
    
    print(f"   Config 1 (10% attack): {key1}")
    print(f"   Config 2 (30% attack): {key2}")
    print(f"   ✅ Keys are unique: {key1 != key2}")
    
    # Test 2: Different spatial radii
    print("\n2. Testing different spatial radii:")
    config3 = GeneratorContextProvider()
    config3.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config3.add('ConnectedDrivingLargeDataCleaner.max_dist', 2000)  # 2km
    config3.add('attack_type', 'ConstOffsetPerID')
    
    config4 = GeneratorContextProvider()
    config4.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config4.add('ConnectedDrivingLargeDataCleaner.max_dist', 200000)  # 200km
    config4.add('attack_type', 'ConstOffsetPerID')
    
    key3 = create_deterministic_cache_key('clean_data', ['test'], config3)
    key4 = create_deterministic_cache_key('clean_data', ['test'], config4)
    
    print(f"   Config 3 contexts: {config3.getAll()}")
    print(f"   Config 4 contexts: {config4.getAll()}")
    print(f"   Config 3 (2km radius):   {key3}")
    print(f"   Config 4 (200km radius): {key4}")
    print(f"   ✅ Keys are unique: {key3 != key4}")
    
    # Test 3: Different attack distance ranges
    print("\n3. Testing different attack distance ranges:")
    config5 = GeneratorContextProvider()
    config5.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config5.add('ConnectedDrivingAttacker.min_dist', 100)
    config5.add('ConnectedDrivingAttacker.max_dist', 200)
    config5.add('attack_type', 'ConstOffsetPerID')
    
    config6 = GeneratorContextProvider()
    config6.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config6.add('ConnectedDrivingAttacker.min_dist', 50)
    config6.add('ConnectedDrivingAttacker.max_dist', 100)
    config6.add('attack_type', 'ConstOffsetPerID')
    
    key5 = create_deterministic_cache_key('clean_data', ['test'], config5)
    key6 = create_deterministic_cache_key('clean_data', ['test'], config6)
    
    print(f"   Config 5 contexts: {config5.getAll()}")
    print(f"   Config 6 contexts: {config6.getAll()}")
    print(f"   Config 5 (100-200m):     {key5}")
    print(f"   Config 6 (50-100m):      {key6}")
    print(f"   ✅ Keys are unique: {key5 != key6}")
    
    # Test 4: Same configuration should produce same key (reproducibility)
    print("\n4. Testing reproducibility (same config = same key):")
    config7 = GeneratorContextProvider()
    config7.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config7.add('ConnectedDrivingLargeDataCleaner.max_dist', 2000)
    config7.add('attack_type', 'ConstOffsetPerID')
    
    config8 = GeneratorContextProvider()
    config8.add('ConnectedDrivingAttacker.attack_ratio', 0.3)
    config8.add('ConnectedDrivingLargeDataCleaner.max_dist', 2000)
    config8.add('attack_type', 'ConstOffsetPerID')
    
    key7 = create_deterministic_cache_key('clean_data', ['test'], config7)
    key8 = create_deterministic_cache_key('clean_data', ['test'], config8)
    
    print(f"   Config 7 (identical):    {key7}")
    print(f"   Config 8 (identical):    {key8}")
    print(f"   ✅ Keys are identical: {key7 == key8}")
    
    # Test 5: No context provider (backward compatibility)
    print("\n5. Testing backward compatibility (no context provider):")
    key9 = create_deterministic_cache_key('clean_data', ['test'], None)
    key10 = create_deterministic_cache_key('clean_data', ['test'], None)
    
    print(f"   No context 1:            {key9}")
    print(f"   No context 2:            {key10}")
    print(f"   ✅ Keys are identical: {key9 == key10}")
    
    print("\n" + "=" * 60)
    
    # Summary
    all_unique_keys = {key1, key2, key3, key4, key5, key6, key9}
    expected_unique_count = 6  # key7==key8 and key9==key10 are expected duplicates
    
    print(f"🔍 SUMMARY:")
    print(f"   Total unique keys generated: {len(all_unique_keys)}")
    print(f"   Expected unique keys: {expected_unique_count}")
    print(f"   ✅ Cache collision prevention: {'SUCCESS' if len(all_unique_keys) == expected_unique_count else 'FAILED'}")
    
    if len(all_unique_keys) == expected_unique_count:
        print("   🎉 All tests passed - cache key collisions have been eliminated!")
    else:
        print("   ⚠️  Some configurations are still producing identical cache keys!")
        print("       This indicates cache collision issues still exist.")

if __name__ == "__main__":
    test_cache_key_uniqueness()