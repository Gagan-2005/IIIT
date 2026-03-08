"""
Native Language Identification with Cuisine Recommendation
Complete inference pipeline for predicting native language and suggesting cuisine
"""

import json
import joblib
import numpy as np
from pathlib import Path
import sys

# Import feature extraction
from extract_hubert_features import extract_hubert_features
from persistent_hubert import extract_live_features, init_hubert
import joblib as _joblib
_PAIRWISE_CACHE = {}

def _load_pairwise_verifier():
    """Load Andhra vs Jharkhand verifier artifacts if present (cached)."""
    global _PAIRWISE_CACHE
    if _PAIRWISE_CACHE:
        return _PAIRWISE_CACHE
    base = Path("models")
    model_p = base / "andhra_jharkhand_verifier.joblib"
    scaler_p = base / "andhra_jharkhand_scaler.joblib"
    info_p = base / "andhra_jharkhand_info.json"
    pca_p = base / "andhra_jharkhand_pca.joblib"
    if not model_p.exists() or not scaler_p.exists() or not info_p.exists():
        return None
    try:
        with open(info_p, 'r') as f:
            info = json.load(f)
        model = _joblib.load(model_p)
        scaler = _joblib.load(scaler_p)
        pca = _joblib.load(pca_p) if pca_p.exists() else None
        _PAIRWISE_CACHE = {'model': model, 'scaler': scaler, 'pca': pca, 'info': info}
        return _PAIRWISE_CACHE
    except Exception:
        return None

# Configuration
MODELS_DIR = Path("models")
BEST_LAYER = 3
CUISINE_MAPPING_PATH = "cuisine_mapping.json"

def load_models():
    """Load all required models and preprocessors"""
    print("Loading models...")
    
    model_path = MODELS_DIR / "speaker_normalized" / "rf_hubert.joblib"
    scaler_path = MODELS_DIR / "speaker_normalized" / "scaler.joblib"
    pca_path = MODELS_DIR / "speaker_normalized" / "pca.joblib"
    le_path = MODELS_DIR / "speaker_normalized" / "label_encoder.joblib"
    
    # Check if models exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease run scripts/train_speaker_normalized.py first!")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    pca = None
    if pca_path.exists():
        pca = joblib.load(pca_path)
        print("  - Loaded: Model, Scaler, PCA, Label Encoder")
    else:
        print("  - Loaded: Model, Scaler, Label Encoder")
    
    return model, scaler, pca, le

def load_cuisine_mapping():
    """Load cuisine recommendations mapping"""
    with open(CUISINE_MAPPING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def _apply_live_calibration(layer_vec: np.ndarray) -> np.ndarray:
    """Calibrate live Layer 3 vector if calibration file exists."""
    calib_path = Path("models/live_calibration.npz")
    if not calib_path.exists():
        return layer_vec
    try:
        data = np.load(calib_path)
        mean_cached = data['mean_cached']
        std_cached = data['std_cached']
        mean_live = data['mean_live']
        std_live = data['std_live']
        norm = (layer_vec - mean_live) / std_live
        mapped = norm * std_cached + mean_cached
        return mapped
    except Exception:
        return layer_vec

def predict_native_language(audio_path, model, scaler, pca, le, device='cpu', allow_live=True):
    """
    Predict native language from audio file
    
    Args:
        audio_path: Path to audio file
        model: Trained classifier
        scaler: Feature scaler
        pca: PCA transformer (optional)
        le: Label encoder
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*60}")
    print("PROCESSING AUDIO")
    print(f"{'='*60}")
    
    # Try to use pre-extracted features first (for consistency with training)
    audio_path_obj = Path(audio_path)
    audio_filename = audio_path_obj.stem
    preextracted_path = Path("features/hubert") / f"{audio_filename}.npz"
    
    if preextracted_path.exists():
        print(f"Using pre-extracted features: {preextracted_path.name}")
        try:
            data = np.load(preextracted_path)
            pooled_features = data['pooled']
            print("  - Loaded from cache")
        except Exception as e:
            print(f"  - Warning: Could not load pre-extracted features: {e}")
            print("  - Extracting and saving features...")
            features_dict = extract_hubert_features(audio_path, output_path=str(preextracted_path), device=device)
            # Load saved features to ensure consistency
            data = np.load(preextracted_path)
            pooled_features = data['pooled']
    else:
        if allow_live:
            print("No cached features; using persistent live extraction")
            init_hubert()
            pooled_features = extract_live_features(audio_path)
        else:
            print("No pre-extracted features found and live disabled.")
            features_dict = extract_hubert_features(audio_path, output_path=None, device=device)
            pooled_features = features_dict['pooled']
    
    # Extract best layer features
    if pooled_features.shape[0] <= BEST_LAYER:
        raise ValueError(f"Expected at least {BEST_LAYER+1} layers, got {pooled_features.shape[0]}")
    
    layer_features = pooled_features[BEST_LAYER]
    print(f"\nUsing Layer {BEST_LAYER} features: {layer_features.shape}")
    
    # Preprocess
    # Calibrate if live extraction was used (no cache)
    if not preextracted_path.exists() and allow_live:
        layer_features = _apply_live_calibration(layer_features)

    layer_features = layer_features.reshape(1, -1)
    layer_features_scaled = scaler.transform(layer_features)
    
    if pca is not None:
        layer_features_pca = pca.transform(layer_features_scaled)
    else:
        layer_features_pca = layer_features_scaled
    
    # Predict base model
    prediction = model.predict(layer_features_pca)[0]
    probabilities = model.predict_proba(layer_features_pca)[0]
    
    # Get label
    predicted_label = le.inverse_transform([prediction])[0]
    
    # Get all class probabilities
    all_classes = le.classes_
    class_probs = {cls: float(prob) for cls, prob in zip(all_classes, probabilities)}
    
    # Sort by probability
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    
    override_applied = False
    # Optional pairwise override for Andhra vs Jharkhand (relaxed thresholds)
    if 'andhra_pradesh' in class_probs and 'jharkhand' in class_probs:
        base_j = class_probs['jharkhand']
        base_a = class_probs['andhra_pradesh']
        if predicted_label == 'jharkhand' and base_j < 0.70 and base_a > 0.02:
            pairwise = _load_pairwise_verifier()
            if pairwise:
                multilayer_vec = None
                audio_filename = audio_path_obj.stem
                preextracted_path = Path("features/hubert") / f"{audio_filename}.npz"
                try:
                    if preextracted_path.exists():
                        data_full = np.load(preextracted_path)
                        if 'pooled' in data_full and data_full['pooled'].shape[0] >= 13:
                            multilayer_vec = data_full['pooled'].reshape(-1)
                except Exception:
                    pass
                if multilayer_vec is None:
                    multilayer_vec = np.tile(layer_features, 13)
                vec = multilayer_vec.reshape(1, -1)
                vec_s = pairwise['scaler'].transform(vec)
                vec_p = pairwise['pca'].transform(vec_s) if pairwise['pca'] is not None else vec_s
                pw_probs = pairwise['model'].predict_proba(vec_p)[0]
                classes_pw = pairwise['info']['classes']
                prob_andhra = pw_probs[classes_pw.index('andhra_pradesh')] if 'andhra_pradesh' in classes_pw else 0.0
                prob_jharkhand = pw_probs[classes_pw.index('jharkhand')] if 'jharkhand' in classes_pw else 0.0
                blended_andhra = 0.6 * prob_andhra + 0.4 * base_a
                blended_jharkhand = 0.6 * prob_jharkhand + 0.4 * base_j
                # Override criteria (relaxed):
                if (
                    (prob_andhra > 0.50 and prob_andhra - prob_jharkhand > 0.05) or
                    (blended_andhra > blended_jharkhand and blended_andhra > 0.35)
                ):
                    predicted_label = 'andhra_pradesh'
                    override_applied = True
                    print(f"  ⚡ Pairwise override: Andhra {prob_andhra:.3f} vs Jharkhand {prob_jharkhand:.3f}")
    return {
        'predicted_label': predicted_label,
        'confidence': float(probabilities[prediction]),
        'all_probabilities': class_probs,
        'top_3': sorted_probs[:3],
        'override_applied': override_applied
    }

def get_cuisine_recommendation(predicted_label, cuisine_mapping):
    """Get cuisine recommendation based on predicted native language"""
    if predicted_label not in cuisine_mapping:
        return None
    
    info = cuisine_mapping[predicted_label]
    return info

def format_output(prediction_result, cuisine_info):
    """Format prediction and cuisine recommendation for display"""
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    
    confidence = prediction_result['confidence']
    confidence_indicator = "✓" if confidence >= 0.60 else "⚠️"
    
    print(f"\n🎯 PREDICTED NATIVE LANGUAGE: {prediction_result['predicted_label'].upper()} {confidence_indicator}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    if confidence < 0.60:
        print("   ⚠️  Low confidence - consider top 2 predictions:")
        top_2 = prediction_result['top_3'][:2]
        print(f"      1. {top_2[0][0]}: {top_2[0][1]*100:.2f}%")
        print(f"      2. {top_2[1][0]}: {top_2[1][1]*100:.2f}%")
    
    print("\n📊 TOP 3 PREDICTIONS:")
    for i, (label, prob) in enumerate(prediction_result['top_3'], 1):
        bar_length = int(prob * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"   {i}. {label:20s} {bar} {prob*100:.2f}%")
    
    print(f"\n{'='*60}")
    print("CUISINE RECOMMENDATION")
    print(f"{'='*60}")
    
    if cuisine_info:
        print(f"\n🍽️  REGION: {cuisine_info['region']}")
        print(f"🗣️  LANGUAGE: {cuisine_info['language']}")
        print(f"🥘 CUISINE: {cuisine_info['cuisine']}")
        
        print("\n📜 MUST-TRY DISHES:")
        for i, dish in enumerate(cuisine_info['dishes'], 1):
            print(f"   {i}. {dish}")
        
        print("\n✨ CHARACTERISTICS:")
        for char in cuisine_info['characteristics']:
            print(f"   • {char}")
    else:
        print("\n⚠️  No cuisine information available for this region.")
    
    print(f"\n{'='*60}\n")

def main():
    if len(sys.argv) < 2:
        print("="*60)
        print("Native Language Identification with Cuisine Recommendation")
        print("="*60)
        print("\nUsage: python predict.py <audio_path>")
        print("Example: python predict.py test_audio.wav")
        print("\nSupported formats: WAV, MP3, FLAC, etc.")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Check device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    try:
        # Load models
        model, scaler, pca, le = load_models()
        
        # Load cuisine mapping
        cuisine_mapping = load_cuisine_mapping()
        
        # Predict
        prediction_result = predict_native_language(
            audio_path, model, scaler, pca, le, device
        )
        
        # Get cuisine recommendation
        cuisine_info = get_cuisine_recommendation(
            prediction_result['predicted_label'],
            cuisine_mapping
        )
        
        # Display results
        format_output(prediction_result, cuisine_info)
        
        # Save results to JSON
        output_file = Path(audio_path).stem + "_prediction.json"
        output_data = {
            'audio_file': str(audio_path),
            'prediction': prediction_result,
            'cuisine_recommendation': cuisine_info
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
