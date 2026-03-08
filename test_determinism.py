"""
Test if HuBERT model gives deterministic outputs
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from extract_hubert_features import extract_hubert_features

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"

print("="*70)
print("DETERMINISM TEST")
print("="*70)

# Extract features twice with same seed
print("\n1️⃣ First extraction (seed=42):")
set_seed(42)
result1 = extract_hubert_features(audio_path, output_path=None, device='cpu')
features1 = result1['pooled'][3]
print(f"   Mean: {features1.mean():.10f}")
print(f"   First 5 values: {features1[:5]}")

print("\n2️⃣ Second extraction (seed=42):")
set_seed(42)
result2 = extract_hubert_features(audio_path, output_path=None, device='cpu')
features2 = result2['pooled'][3]
print(f"   Mean: {features2.mean():.10f}")
print(f"   First 5 values: {features2[:5]}")

print("\n🔍 COMPARISON:")
if np.allclose(features1, features2):
    print("   ✅ DETERMINISTIC! Same seed gives same features")
else:
    diff = np.abs(features1 - features2).mean()
    print(f"   ❌ NON-DETERMINISTIC! Mean difference: {diff:.10f}")
    print("   This means model has randomness even with fixed seed")

# Now compare with cached
print("\n3️⃣ Cached features:")
cached = np.load("features/hubert/Andhra_speaker (1083).npz")
cached_features = cached['pooled'][3]
print(f"   Mean: {cached_features.mean():.10f}")
print(f"   First 5 values: {cached_features[:5]}")

print("\n🔍 CACHED vs REAL-TIME:")
if np.allclose(features1, cached_features):
    print("   ✅ IDENTICAL!")
else:
    diff = np.abs(features1 - cached_features).mean()
    print(f"   ❌ DIFFERENT! Mean difference: {diff:.10f}")
