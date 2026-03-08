"""Debug predict_speaker_normalized.py"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import numpy as np
import joblib

# Load models the way predict_speaker_normalized does
MODELS_DIR = Path("models/speaker_normalized")

model = joblib.load(MODELS_DIR / "rf_hubert.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
pca = joblib.load(MODELS_DIR / "pca.joblib")
le = joblib.load(MODELS_DIR / "label_encoder.joblib")

print("Loaded speaker-normalized models")
print(f"  - Scaler: {scaler.n_features_in_} features")
print(f"  - PCA: {pca.n_features_in_} → {pca.n_components_} features")
print(f"  - Model: {model.n_features_in_} features")

# Load cached features
cached_path = Path("features/hubert/Tamil_speaker (1).npz")
data = np.load(cached_path, allow_pickle=True)
pooled = data["pooled"]

print(f"\nCached features:")
print(f"  - Pooled shape: {pooled.shape}")

# Extract layer 3 - this is key!
BEST_LAYER = 3

# Check if pooled is already averaged or still has time dimension
if pooled.ndim == 2:
    # pooled is (n_layers, n_features) - already averaged
    layer_features = pooled[BEST_LAYER]
    print(f"  - Already pooled/averaged: {layer_features.shape}")
else:
    # pooled is (n_layers, time_steps, n_features) - need to average
    layer_features = pooled[BEST_LAYER].mean(axis=0)
    print(f"  - Averaging time dimension: {layer_features.shape}")

# Reshape for sklearn
features_normalized = layer_features.reshape(1, -1)
print(f"\nLayer {BEST_LAYER} features:")
print(f"  - Shape: {features_normalized.shape}")
print(f"  - First 5 values: {features_normalized[0, :5]}")

# Apply transformations
features_scaled = scaler.transform(features_normalized)
print(f"\nAfter scaler:")
print(f"  - Shape: {features_scaled.shape}")
print(f"  - First 5 values: {features_scaled[0, :5]}")

features_pca = pca.transform(features_scaled)
print(f"\nAfter PCA:")
print(f"  - Shape: {features_pca.shape}")
print(f"  - First 5 values: {features_pca[0, :5]}")

# Predict
probabilities = model.predict_proba(features_pca)[0]
prediction = np.argmax(probabilities)

print(f"\nPrediction:")
print(f"  - Predicted: {le.classes_[prediction]}")
print(f"  - Confidence: {probabilities[prediction]*100:.2f}%")
print(f"\n  Top 3:")
top_3_idx = np.argsort(probabilities)[::-1][:3]
for i, idx in enumerate(top_3_idx, 1):
    print(f"    {i}. {le.classes_[idx]}: {probabilities[idx]*100:.2f}%")
