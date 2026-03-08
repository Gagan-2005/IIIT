"""Test averaging all layers vs using layer 3"""
import numpy as np
from pathlib import Path
import joblib

# Load models
MODELS_DIR = Path("models/speaker_normalized")
model = joblib.load(MODELS_DIR / "rf_hubert.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
pca = joblib.load(MODELS_DIR / "pca.joblib")
le = joblib.load(MODELS_DIR / "label_encoder.joblib")

# Load cached features
cached_path = Path("features/hubert/Tamil_speaker (1).npz")
data = np.load(cached_path, allow_pickle=True)
pooled = data['pooled']  # Shape: (26, 768)

print("Testing different feature extraction methods:\n")
print("="*70)

# Method 1: Layer 3 only (what training used)
layer3_features = pooled[3].reshape(1, -1)
layer3_scaled = scaler.transform(layer3_features)
layer3_pca = pca.transform(layer3_scaled)
layer3_probs = model.predict_proba(layer3_pca)[0]
layer3_pred = np.argmax(layer3_probs)

print("METHOD 1: Layer 3 only (CORRECT - matches training)")
print(f"  Predicted: {le.classes_[layer3_pred]}")
print(f"  Confidence: {layer3_probs[layer3_pred]*100:.2f}%")
print(f"  Top 3:")
for idx in np.argsort(layer3_probs)[::-1][:3]:
    print(f"    - {le.classes_[idx]}: {layer3_probs[idx]*100:.2f}%")

print("\n" + "="*70)

# Method 2: Average all 26 layers (what predict_speaker_normalized.py does by mistake)
avg_features = pooled.mean(axis=0).reshape(1, -1)
avg_scaled = scaler.transform(avg_features)
avg_pca = pca.transform(avg_scaled)
avg_probs = model.predict_proba(avg_pca)[0]
avg_pred = np.argmax(avg_probs)

print("METHOD 2: Average all 26 layers (WRONG - but used by predict_speaker_normalized.py)")
print(f"  Predicted: {le.classes_[avg_pred]}")
print(f"  Confidence: {avg_probs[avg_pred]*100:.2f}%")
print(f"  Top 3:")
for idx in np.argsort(avg_probs)[::-1][:3]:
    print(f"    - {le.classes_[idx]}: {avg_probs[idx]*100:.2f}%")

print("\n" + "="*70)
print("\nCONCLUSION:")
print("The script predict_speaker_normalized.py has a BUG!")
print("It averages all 26 layers instead of using Layer 3 only.")
print("This accidentally gives better predictions but is WRONG.")
