"""Detailed debugging of predict_backend"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import numpy as np

# Manually replicate what predict_backend does
from predict import load_models

print("Loading models...")
model, scaler, pca, le = load_models()

print(f"Loaded models:")
print(f"  - Scaler: {scaler.n_features_in_} input features")
print(f"  - PCA: {pca.n_features_in_} → {pca.n_components_} features")
print(f"  - Model: {model.n_features_in_} input features")
print(f"  - Classes: {list(le.classes_)}")

# Load cached features for Tamil speaker (1)
cached_path = Path("features/hubert/Tamil_speaker (1).npz")
data = np.load(cached_path, allow_pickle=True)
pooled = data["pooled"]

print(f"\nLoaded cached features:")
print(f"  - Pooled shape: {pooled.shape}")
print(f"  - Type: {type(pooled)}")

# Extract layer 3
BEST_LAYER = 3
layer_vec = pooled[BEST_LAYER].reshape(1, -1)
print(f"\nLayer {BEST_LAYER} features:")
print(f"  - Shape: {layer_vec.shape}")
print(f"  - First 5 values: {layer_vec[0, :5]}")

# Apply scaler
layer_scaled = scaler.transform(layer_vec)
print(f"\nAfter scaler:")
print(f"  - Shape: {layer_scaled.shape}")
print(f"  - First 5 values: {layer_scaled[0, :5]}")

# Apply PCA
layer_pca = pca.transform(layer_scaled)
print(f"\nAfter PCA:")
print(f"  - Shape: {layer_pca.shape}")
print(f"  - First 5 values: {layer_pca[0, :5]}")

# Predict
probs = model.predict_proba(layer_pca)[0]
pred_idx = np.argmax(probs)
pred_label = le.classes_[pred_idx]

print(f"\nPrediction:")
print(f"  - Predicted: {pred_label}")
print(f"  - Confidence: {probs[pred_idx]*100:.2f}%")
print(f"\n  Top 3:")
top_indices = np.argsort(probs)[::-1][:3]
for i, idx in enumerate(top_indices, 1):
    print(f"    {i}. {le.classes_[idx]}: {probs[idx]*100:.2f}%")
