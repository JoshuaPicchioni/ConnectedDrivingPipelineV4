#!/usr/bin/env python3
"""
Simple isolated test for cache key generation
"""

import sys
sys.path.append('/home/ubuntu/repos/ConnectedDrivingPipelineV4')

from Decorators.FileCache import create_deterministic_cache_key
from ServiceProviders.GeneratorContextProvider import GeneratorContextProvider

# Test 1: Create two completely separate provider instances
print("=== Test 1: Separate instances ===")

provider_a = GeneratorContextProvider()
provider_a.add('attack_ratio', 0.1)
print(f"Provider A after setting 0.1: {provider_a.getAll()}")

provider_b = GeneratorContextProvider()
provider_b.add('attack_ratio', 0.3)
print(f"Provider B after setting 0.3: {provider_b.getAll()}")

# Check if they're independent
print(f"Provider A final: {provider_a.getAll()}")
print(f"Provider B final: {provider_b.getAll()}")

# Test 2: Generate cache keys
key_a = create_deterministic_cache_key('test_func', ['var1'], provider_a)
key_b = create_deterministic_cache_key('test_func', ['var1'], provider_b)

print(f"\nKey A: {key_a}")
print(f"Key B: {key_b}")
print(f"Keys different: {key_a != key_b}")

# Test 3: Create providers with dict initialization
print("\n=== Test 3: Dict initialization ===")
provider_c = GeneratorContextProvider({'attack_ratio': 0.1})
provider_d = GeneratorContextProvider({'attack_ratio': 0.3})

print(f"Provider C: {provider_c.getAll()}")
print(f"Provider D: {provider_d.getAll()}")

key_c = create_deterministic_cache_key('test_func', ['var1'], provider_c)
key_d = create_deterministic_cache_key('test_func', ['var1'], provider_d)

print(f"\nKey C: {key_c}")
print(f"Key D: {key_d}")
print(f"Keys different: {key_c != key_d}")