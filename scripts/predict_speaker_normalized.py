"""
Prediction Script for Speaker-Normalized Model
Uses speaker normalization approach for consistent inference
"""

import sys
import joblib
import numpy as np
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from persistent_hubert import init_hubert, extract_live_features

# Configuration
MODELS_DIR = Path("models/speaker_normalized")
BEST_LAYER = 3


def load_models():
    """Load all trained models"""
    print("Loading speaker-normalized models...")
    
    model = joblib.load(MODELS_DIR / "rf_hubert.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    pca = joblib.load(MODELS_DIR / "pca.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    
    print("  - Loaded: Model, Scaler, PCA, Label Encoder")
    return model, scaler, pca, le


def extract_features(audio_path, device='cpu'):
    """Extract HuBERT features - try cached first, then live extraction"""
    print("\n" + "="*60)
    print("PROCESSING AUDIO (SPEAKER-NORMALIZED)")
    print("="*60)
    
    # Try to use cached features first (matches training exactly)
    audio_name = Path(audio_path).stem
    cached_path = Path("features/hubert") / f"{audio_name}.npz"
    
    if cached_path.exists():
        print(f"Using cached features: {cached_path.name}")
        data = np.load(cached_path)
        if 'pooled' in data:
            # Cached features are (n_frames, 768), average to get (768,)
            features = data['pooled'].mean(axis=0)
            print(f"Loaded cached features: {features.shape}")
            return features
    
    # Fallback to live extraction
    print("Extracting HuBERT features (live)...")
    init_hubert(device=device)
    features = extract_live_features(audio_path)  # Returns (13, 768)
    
    if features is None:
        raise ValueError(f"Failed to extract features from {audio_path}")
    
    # Get layer 3 (features already pooled/averaged by persistent_hubert)
    layer_features = features[BEST_LAYER]  # Shape: (768,)
    print(f"\nUsing Layer {BEST_LAYER} features (live): {layer_features.shape}")
    
    return layer_features


def predict_native_language(audio_path, model, scaler, pca, le, device='cpu'):
    """
    Predict native language with speaker normalization approach
    
    NOTE: For single-sample prediction, we cannot apply per-speaker normalization
    since we need multiple samples from the same speaker to compute mean/std.
    
    The model was trained with speaker normalization, so predictions on single
    samples may be less accurate than on multiple samples from the same speaker.
    """
    # Extract features
    features = extract_features(audio_path, device)
    
    # For single sample, we skip speaker normalization
    # (would need multiple samples from same speaker)
    features_normalized = features.reshape(1, -1)
    
    # Apply global standardization
    features_scaled = scaler.transform(features_normalized)
    
    # Apply PCA
    features_pca = pca.transform(features_scaled)
    
    # Predict
    prediction = model.predict(features_pca)[0]
    probabilities = model.predict_proba(features_pca)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(probabilities)[::-1][:3]
    top_3 = [(le.classes_[idx], probabilities[idx]) for idx in top_3_idx]
    
    result = {
        'predicted_label': le.classes_[prediction],
        'confidence': float(probabilities[prediction]),
        'probabilities': {label: float(prob) for label, prob in zip(le.classes_, probabilities)},
        'top_3': top_3
    }
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_speaker_normalized.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Check if models exist
    if not MODELS_DIR.exists():
        print(f"Error: Models directory not found: {MODELS_DIR}")
        print("\nPlease train the model first:")
        print("  python scripts/train_speaker_normalized.py")
        sys.exit(1)
    
    # Load models
    model, scaler, pca, le = load_models()
    
    # Predict
    result = predict_native_language(audio_file, model, scaler, pca, le, device='cpu')
    
    # Display result
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\n✓ Predicted Region: {result['predicted_label']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"\n  Top 3 Predictions:")
    for i, (label, prob) in enumerate(result['top_3'], 1):
        print(f"    {i}. {label}: {prob*100:.2f}%")
    
    print("\n" + "="*60)
    print("\nNOTE: This model uses speaker normalization.")
    print("For best results, provide multiple samples from the same speaker.")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    main()
