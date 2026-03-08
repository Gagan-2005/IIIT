"""Debug: Check PCA dimensions"""
import joblib
from pathlib import Path
import numpy as np

# Check speaker-normalized models
model_path = Path("models/speaker_normalized/rf_hubert.joblib")
scaler_path = Path("models/speaker_normalized/scaler.joblib")
pca_path = Path("models/speaker_normalized/pca.joblib")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)

print("Speaker-Normalized Models:")
print(f"  - PCA input dims: {pca.n_features_in_}")
print(f"  - PCA output dims: {pca.n_components_}")
print(f"  - Scaler input dims: {scaler.n_features_in_}")
print(f"  - Model input dims: {model.n_features_in_}")
print(f"  - Expected pipeline: 768 → scaler → PCA → 96 → model")

# Test the pipeline
dummy_features = np.random.randn(1, 768)
print(f"\nTest pipeline:")
print(f"  Raw features: {dummy_features.shape}")

scaled = scaler.transform(dummy_features)
print(f"  After scaler: {scaled.shape}")

pca_transformed = pca.transform(scaled)
print(f"  After PCA: {pca_transformed.shape}")

pred = model.predict(pca_transformed)
print(f"  Prediction shape: {pred.shape}")
print(f"\n✓ Pipeline works correctly!")
