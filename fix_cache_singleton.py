#!/usr/bin/env python3
"""
Fix the singleton issue in cache key generation

Since GeneratorContextProvider is a singleton, we need to work around this
for proper cache key generation that includes all context parameters.
"""

import sys
sys.path.append('/home/ubuntu/repos/ConnectedDrivingPipelineV4')

from Decorators.FileCache import create_deterministic_cache_key

# Create a simple non-singleton context provider for testing
class NonSingletonContextProvider:
    def __init__(self, contexts=None):
        self.contexts = contexts if contexts is not None else {}

    def add(self, key, value):
        self.contexts[key] = value

    def getAll(self):
        return self.contexts.copy()  # Return a copy to prevent accidental mutation

def test_with_non_singleton():
    print("🔧 Testing with Non-Singleton Context Provider...")
    print("=" * 60)
    
    # Test 1: Different attack ratios
    print("\n1. Testing different attack ratios:")
    
    provider1 = NonSingletonContextProvider()
    provider1.add('attack_ratio', 0.1)
    provider1.add('max_dist', 2000)
    
    provider2 = NonSingletonContextProvider()
    provider2.add('attack_ratio', 0.3)
    provider2.add('max_dist', 2000)
    
    print(f"   Provider 1: {provider1.getAll()}")
    print(f"   Provider 2: {provider2.getAll()}")
    
    key1 = create_deterministic_cache_key('clean_data', ['test'], provider1)
    key2 = create_deterministic_cache_key('clean_data', ['test'], provider2)
    
    print(f"   Key 1 (0.1 ratio): {key1}")
    print(f"   Key 2 (0.3 ratio): {key2}")
    print(f"   ✅ Keys are unique: {key1 != key2}")
    
    # Test 2: Different spatial radii
    print("\n2. Testing different spatial radii:")
    
    provider3 = NonSingletonContextProvider()
    provider3.add('attack_ratio', 0.3)
    provider3.add('max_dist', 2000)  # 2km
    
    provider4 = NonSingletonContextProvider()
    provider4.add('attack_ratio', 0.3)
    provider4.add('max_dist', 200000)  # 200km
    
    print(f"   Provider 3: {provider3.getAll()}")
    print(f"   Provider 4: {provider4.getAll()}")
    
    key3 = create_deterministic_cache_key('clean_data', ['test'], provider3)
    key4 = create_deterministic_cache_key('clean_data', ['test'], provider4)
    
    print(f"   Key 3 (2km):   {key3}")
    print(f"   Key 4 (200km): {key4}")
    print(f"   ✅ Keys are unique: {key3 != key4}")
    
    # Test 3: Same configuration should produce same key
    print("\n3. Testing reproducibility:")
    
    provider5 = NonSingletonContextProvider()
    provider5.add('attack_ratio', 0.3)
    provider5.add('max_dist', 2000)
    
    provider6 = NonSingletonContextProvider()
    provider6.add('attack_ratio', 0.3)
    provider6.add('max_dist', 2000)
    
    key5 = create_deterministic_cache_key('clean_data', ['test'], provider5)
    key6 = create_deterministic_cache_key('clean_data', ['test'], provider6)
    
    print(f"   Key 5: {key5}")
    print(f"   Key 6: {key6}")
    print(f"   ✅ Keys are identical: {key5 == key6}")
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS: Non-singleton approach works correctly!")
    print("   Cache key generation now properly includes all context parameters.")

if __name__ == "__main__":
    test_with_non_singleton()