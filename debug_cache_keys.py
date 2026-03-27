#!/usr/bin/env python3
"""
Debug Cache Key Generation

This script helps debug why different configurations are producing identical cache keys.
"""

import sys
import os

# Add the repo directory to path so we can import the modules
sys.path.append('/home/ubuntu/repos/ConnectedDrivingPipelineV4')

from Decorators.FileCache import create_deterministic_cache_key
from ServiceProviders.GeneratorContextProvider import GeneratorContextProvider

def debug_cache_key_generation():
    """Debug the cache key generation process."""
    
    print("🔍 Debugging Cache Key Generation...")
    print("=" * 60)
    
    # Create two different configurations
    config1 = GeneratorContextProvider({
        'ConnectedDrivingAttacker.attack_ratio': 0.1,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID'
    })
    
    config2 = GeneratorContextProvider({
        'ConnectedDrivingAttacker.attack_ratio': 0.3,
        'ConnectedDrivingLargeDataCleaner.max_dist': 2000,
        'attack_type': 'ConstOffsetPerID'
    })
    
    print("Config 1 contexts:", config1.getAll())
    print("Config 2 contexts:", config2.getAll())
    
    # Test if contexts are properly accessible
    print("\nTesting context provider access:")
    print("Config 1 has getAll():", hasattr(config1, 'getAll'))
    print("Config 2 has getAll():", hasattr(config2, 'getAll'))
    
    # Let's manually test the cache key generation step by step
    print("\nManual cache key generation:")
    
    function_name = 'clean_data'
    cache_variables = ['test']
    
    # Start with function name
    key_parts = [function_name]
    print(f"1. Function name: {key_parts}")
    
    # Add cache variables
    str_vars = []
    for var in cache_variables:
        str_vars.append(str(var))
    key_parts.extend(str_vars)
    print(f"2. After cache variables: {key_parts}")
    
    # Add context for config1
    if config1 and hasattr(config1, 'getAll'):
        all_contexts = config1.getAll()
        print(f"3. Config1 contexts: {all_contexts}")
        if all_contexts:
            sorted_contexts = sorted(all_contexts.items()) if isinstance(all_contexts, dict) else str(all_contexts)
            context_part = f"CONTEXT_{str(sorted_contexts)}"
            print(f"4. Config1 context part: {context_part}")
            key_parts_1 = key_parts + [context_part]
        else:
            key_parts_1 = key_parts
    else:
        key_parts_1 = key_parts
        
    print(f"5. Final key parts for config1: {key_parts_1}")
    
    # Add context for config2
    if config2 and hasattr(config2, 'getAll'):
        all_contexts = config2.getAll()
        print(f"6. Config2 contexts: {all_contexts}")
        if all_contexts:
            sorted_contexts = sorted(all_contexts.items()) if isinstance(all_contexts, dict) else str(all_contexts)
            context_part = f"CONTEXT_{str(sorted_contexts)}"
            print(f"7. Config2 context part: {context_part}")
            key_parts_2 = key_parts + [context_part]
        else:
            key_parts_2 = key_parts
    else:
        key_parts_2 = key_parts
        
    print(f"8. Final key parts for config2: {key_parts_2}")
    
    # Generate final keys
    import hashlib
    key_string_1 = "_".join(key_parts_1)
    key_string_2 = "_".join(key_parts_2)
    
    print(f"\n9. Key string 1: {key_string_1}")
    print(f"10. Key string 2: {key_string_2}")
    
    hash1 = hashlib.md5(key_string_1.encode('utf-8')).hexdigest()
    hash2 = hashlib.md5(key_string_2.encode('utf-8')).hexdigest()
    
    print(f"\n11. Hash 1: {hash1}")
    print(f"12. Hash 2: {hash2}")
    print(f"13. Hashes are different: {hash1 != hash2}")

if __name__ == "__main__":
    debug_cache_key_generation()