"""Inference using multilayer concatenated HuBERT model artifacts.

Loads rf_hubert_multilayer.joblib + scaler_multilayer + optional pca_multilayer
and performs prediction on an input audio file. If cached features exist,
they are used; else persistent live extraction is performed and all 13 layers
concatenated.
"""

import sys
from pathlib import Path
import numpy as np
import joblib
from persistent_hubert import extract_live_features, init_hubert
from extract_hubert_features import extract_hubert_features

MODELS = Path("models")
FEATURES = Path("features/hubert")
CALIB = Path("models/live_calibration.npz")

def load_artifacts():
    clf = joblib.load(MODELS / "rf_hubert_multilayer.joblib")
    scaler = joblib.load(MODELS / "scaler_multilayer.joblib")
    pca_path = MODELS / "pca_multilayer.joblib"
    pca = joblib.load(pca_path) if pca_path.exists() else None
    le = joblib.load(MODELS / "label_encoder_multilayer.joblib")
    calib = np.load(CALIB) if CALIB.exists() else None
    return clf, scaler, pca, le, calib

def apply_calibration(pooled, calib):
    if calib is None:
        return pooled
    mean_cached = calib['mean_cached']
    std_cached = calib['std_cached']
    mean_live = calib['mean_live']
    std_live = calib['std_live']
    # apply per-layer for layer 3 only (others left untouched)
    pooled[3] = ((pooled[3] - mean_live) / std_live) * std_cached + mean_cached
    return pooled

def get_features(audio_path: str):
    stem = Path(audio_path).stem
    npz_path = FEATURES / f"{stem}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        pooled = data['pooled']
    else:
        init_hubert()
        pooled = extract_live_features(audio_path)
    return pooled

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_multilayer.py <audio.wav>")
        return 1
    audio = sys.argv[1]
    if not Path(audio).exists():
        print("Audio not found:", audio)
        return 1
    clf, scaler, pca, le, calib = load_artifacts()
    pooled = get_features(audio)
    pooled = apply_calibration(pooled, calib)
    vec = pooled.reshape(-1).reshape(1, -1)
    vec_s = scaler.transform(vec)
    if pca:
        vec_p = pca.transform(vec_s)
    else:
        vec_p = vec_s
    probs = clf.predict_proba(vec_p)[0]
    pred_idx = clf.predict(vec_p)[0]
    label = le.inverse_transform([pred_idx])[0]
    classes = le.classes_
    top = sorted([(c, float(p)) for c, p in zip(classes, probs)], key=lambda x: x[1], reverse=True)
    print("Predicted:", label)
    print("Top 5:")
    for c, p in top[:5]:
        print(f"  {c:15s} {p*100:6.2f}%")
    return 0

if __name__ == '__main__':
    sys.exit(main())
