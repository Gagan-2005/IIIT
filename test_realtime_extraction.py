"""
Test real-time feature extraction vs cached features
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "scripts")

from extract_hubert_features import extract_hubert_features
from predict import load_models, predict_native_language

# Load models
print("Loading models...")
model, scaler, pca, le = load_models()
print()

# Test with Andhra Pradesh sample
audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"
print(f"Testing with: {audio_path}")
print("="*70)

# Test 1: Using cached features (should work)
print("\n1️⃣ USING CACHED FEATURES:")
result = predict_native_language(audio_path, model, scaler, pca, le, device='cpu')
print(f"   Prediction: {result['predicted_label'].upper()}")
print(f"   Confidence: {result['confidence']*100:.2f}%")
top3_str = [f"{label} ({prob*100:.1f}%)" for label, prob in result['top_3'][:3]]
print(f"   Top 3: {top3_str}")

# Test 2: Force real-time extraction (rename cached file temporarily)
print("\n2️⃣ FORCING REAL-TIME EXTRACTION:")
cached_path = Path("features/hubert/Andhra_speaker (1083).npz")
backup_path = Path("features/hubert/Andhra_speaker (1083).npz.backup")

# Move cached file
if cached_path.exists():
    cached_path.rename(backup_path)
    print("   (Temporarily moved cached file)")

try:
    result2 = predict_native_language(audio_path, model, scaler, pca, le, device='cpu')
    print(f"   Prediction: {result2['predicted_label'].upper()}")
    print(f"   Confidence: {result2['confidence']*100:.2f}%")
    top3_str2 = [f"{label} ({prob*100:.1f}%)" for label, prob in result2['top_3'][:3]]
    print(f"   Top 3: {top3_str2}")
    
    # Compare
    print("\n" + "="*70)
    if result['predicted_label'] == result2['predicted_label']:
        print("✅ SAME PREDICTION with both methods!")
    else:
        print("❌ DIFFERENT PREDICTIONS!")
        print(f"   Cached: {result['predicted_label']} ({result['confidence']*100:.1f}%)")
        print(f"   Real-time: {result2['predicted_label']} ({result2['confidence']*100:.1f}%)")
        print("\n⚠️ THIS IS THE BUG! Real-time extraction gives wrong results.")
    
finally:
    # Restore cached file
    if backup_path.exists():
        backup_path.rename(cached_path)
        print("\n(Restored cached file)")
