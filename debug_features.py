"""
Debug: Compare feature values between cached and real-time extraction
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from extract_hubert_features import extract_hubert_features

# Test audio
audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"
audio_name = Path(audio_path).stem

# Load cached features
cached_path = Path(f"features/hubert/{audio_name}.npz")
cached_data = np.load(cached_path)
cached_features = cached_data['pooled'][3]  # Layer 3

print("="*70)
print("FEATURE COMPARISON")
print("="*70)

# Extract real-time features
print("\nExtracting real-time features...")
realtime_data = extract_hubert_features(audio_path, output_path=None, device='cpu')
realtime_features = realtime_data['pooled'][3]  # Layer 3

print(f"\n📊 CACHED FEATURES (Layer 3):")
print(f"   Shape: {cached_features.shape}")
print(f"   Mean: {cached_features.mean():.10f}")
print(f"   Std: {cached_features.std():.10f}")
print(f"   Min: {cached_features.min():.10f}")
print(f"   Max: {cached_features.max():.10f}")
print(f"   First 10 values: {cached_features[:10]}")

print(f"\n📊 REAL-TIME FEATURES (Layer 3):")
print(f"   Shape: {realtime_features.shape}")
print(f"   Mean: {realtime_features.mean():.10f}")
print(f"   Std: {realtime_features.std():.10f}")
print(f"   Min: {realtime_features.min():.10f}")
print(f"   Max: {realtime_features.max():.10f}")
print(f"   First 10 values: {realtime_features[:10]}")

print(f"\n🔍 DIFFERENCE:")
diff = np.abs(cached_features - realtime_features)
print(f"   Mean absolute difference: {diff.mean():.10f}")
print(f"   Max absolute difference: {diff.max():.10f}")
print(f"   Are they equal? {np.allclose(cached_features, realtime_features, rtol=1e-5, atol=1e-8)}")

if not np.allclose(cached_features, realtime_features, rtol=1e-5, atol=1e-8):
    print(f"\n❌ FEATURES ARE DIFFERENT!")
    print(f"   This explains why predictions differ.")
    print(f"\n   Possible causes:")
    print(f"   1. Different model initialization")
    print(f"   2. Different audio preprocessing")
    print(f"   3. Different random seed in model")
    print(f"   4. Model not in eval() mode properly")
else:
    print(f"\n✅ FEATURES ARE IDENTICAL!")
