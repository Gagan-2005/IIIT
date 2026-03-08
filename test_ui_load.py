"""Test if UI models load correctly"""
import joblib
from pathlib import Path

MODELS_DIR = Path("models")

print("Testing model loading...")

# Try loading speaker-normalized models
model_path = MODELS_DIR / "speaker_normalized" / "rf_hubert.joblib"
scaler_path = MODELS_DIR / "speaker_normalized" / "scaler.joblib"
pca_path = MODELS_DIR / "speaker_normalized" / "pca.joblib"
le_path = MODELS_DIR / "speaker_normalized" / "label_encoder.joblib"

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
le = joblib.load(le_path)

print("✓ All models loaded successfully!")
print(f"  - Model: {type(model).__name__}")
print(f"  - Scaler: {type(scaler).__name__}")
print(f"  - PCA: {type(pca).__name__} ({pca.n_components_} components)")
print(f"  - Label Encoder: {len(le.classes_)} classes")
print(f"  - Classes: {list(le.classes_)}")
print("\nThe UI (app.py) should work properly now! ✓")
