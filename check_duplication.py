import numpy as np
from pathlib import Path

# Load one cached file
file = Path('features/hubert/Andhra_speaker (1084).npz')
data = np.load(file)
pooled = data['pooled']

print(f"Shape: {pooled.shape}")
print(f"\nChecking if data is duplicated...")
print(f"First half shape: {pooled[:13].shape}")
print(f"Second half shape: {pooled[13:].shape}")

# Check if first 13 layers == second 13 layers
if pooled.shape[0] == 26:
    first_half = pooled[:13]
    second_half = pooled[13:]
    
    # Check if they're identical
    are_identical = np.allclose(first_half, second_half, rtol=1e-5, atol=1e-8)
    print(f"\nAre first 13 and second 13 identical? {are_identical}")
    
    if are_identical:
        print("✅ CONFIRMED: Cached features are duplicated!")
        print("   Solution: Use only first 13 layers (pooled[:13])")
    else:
        print("❌ Not identical - data might be from different source")
        print(f"\n   Difference stats:")
        diff = np.abs(first_half - second_half)
        print(f"   Max diff: {diff.max():.6f}")
        print(f"   Mean diff: {diff.mean():.6f}")
        print(f"   Min diff: {diff.min():.6f}")
