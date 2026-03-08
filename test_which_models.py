"""Test which models the app actually loads"""
import sys
import joblib
from pathlib import Path

# Simulate what app.py does
MODELS_DIR = Path("models")

print("="*70)
print("CHECKING WHICH MODELS APP.PY LOADS")
print("="*70)

# Check speaker-normalized models
model_path = MODELS_DIR / "speaker_normalized" / "rf_hubert.joblib"
scaler_path = MODELS_DIR / "speaker_normalized" / "scaler.joblib"
pca_path = MODELS_DIR / "speaker_normalized" / "pca.joblib"
le_path = MODELS_DIR / "speaker_normalized" / "label_encoder.joblib"

print(f"\nApp.py should load from:")
print(f"  Model: {model_path}")
print(f"  Exists: {model_path.exists()}")

if model_path.exists():
    model = joblib.load(model_path)
    print(f"\n  Model type: {type(model).__name__}")
    print(f"  Model features: {model.n_features_in_}")
    print(f"  Model estimators: {model.n_estimators}")

# Now check the predict_backend
sys.path.insert(0, 'scripts')
from predict import load_models

print("\n" + "="*70)
print("CHECKING BACKEND load_models()")
print("="*70)

model2, scaler2, pca2, le2 = load_models()
print(f"  Model features: {model2.n_features_in_}")
print(f"  Model estimators: {model2.n_estimators}")
print(f"  PCA components: {pca2.n_components_}")
print(f"  Classes: {list(le2.classes_)}")

# Test a prediction
print("\n" + "="*70)
print("TESTING PREDICTION WITH BACKEND MODELS")
print("="*70)

import numpy as np
cached_path = Path("features/hubert/Andhra_speaker (1084).npz")
data = np.load(cached_path)
pooled = data['pooled']

# Average all layers
layer_vec = pooled.mean(axis=0).reshape(1, -1)
layer_scaled = scaler2.transform(layer_vec)
layer_pca = pca2.transform(layer_scaled)
probs = model2.predict_proba(layer_pca)[0]
pred_idx = np.argmax(probs)

print(f"\nPrediction: {le2.classes_[pred_idx]}")
print(f"Confidence: {probs[pred_idx]*100:.2f}%")

if le2.classes_[pred_idx] == 'andhra_pradesh':
    print("\n✓ CORRECT - Backend models work!")
else:
    print(f"\n✗ WRONG - Backend is broken!")
