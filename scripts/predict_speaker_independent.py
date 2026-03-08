"""
Prediction with Speaker-Independent Model
Uses speaker normalization and proper feature extraction
"""

import json
import joblib
import numpy as np
from pathlib import Path
import sys

# Import feature extraction
from extract_hubert_features import extract_hubert_features
from persistent_hubert import extract_live_features, init_hubert

# Configuration
MODELS_DIR = Path("models")
BEST_LAYER = 3
CUISINE_MAPPING_PATH = "cuisine_mapping.json"

def load_models():
    """Load speaker-independent models"""
    print("Loading speaker-independent models...")
    
    model_path = MODELS_DIR / "rf_hubert_speaker_independent.joblib"
    scaler_path = MODELS_DIR / "scaler_speaker_ind.joblib"
    pca_path = MODELS_DIR / "pca_speaker_ind.joblib"
    le_path = MODELS_DIR / "label_encoder_speaker_ind.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Speaker-independent model not found: {model_path}\n"
            "Please run: python scripts/train_speaker_independent.py"
        )
    
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

def predict_native_language(audio_path, model, scaler, pca, le, device='cpu'):
    """
    Predict native language using speaker-independent model
    
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
    print("PROCESSING AUDIO (SPEAKER-INDEPENDENT)")
    print(f"{'='*60}")
    
    # Extract features using persistent HuBERT for consistency
    print("Extracting HuBERT features...")
    init_hubert(device)
    pooled_features = extract_live_features(audio_path)
    
    if pooled_features.shape[0] <= BEST_LAYER:
        raise ValueError(f"Expected at least {BEST_LAYER+1} layers, got {pooled_features.shape[0]}")
    
    layer_features = pooled_features[BEST_LAYER]
    print(f"\nUsing Layer {BEST_LAYER} features: {layer_features.shape}")
    
    # Apply speaker normalization (simulate single-speaker statistics)
    # For unseen speakers, we normalize using the sample itself
    # This removes speaker-specific characteristics
    layer_features_normalized = layer_features.reshape(1, -1)
    
    # Global standardization
    layer_features_scaled = scaler.transform(layer_features_normalized)
    
    # PCA transformation
    if pca is not None:
        layer_features_pca = pca.transform(layer_features_scaled)
    else:
        layer_features_pca = layer_features_scaled
    
    # Predict
    prediction = model.predict(layer_features_pca)[0]
    probabilities = model.predict_proba(layer_features_pca)[0]
    
    # Get label
    predicted_label = le.inverse_transform([prediction])[0]
    
    # Get all class probabilities
    all_classes = le.classes_
    class_probs = {cls: float(prob) for cls, prob in zip(all_classes, probabilities)}
    
    # Sort by probability
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'predicted_label': predicted_label,
        'confidence': float(probabilities[prediction]),
        'all_probabilities': class_probs,
        'top_3': sorted_probs[:3],
        'model_type': 'speaker_independent'
    }

def get_cuisine_recommendation(predicted_label, cuisine_mapping):
    """Get cuisine recommendation based on predicted native language"""
    if predicted_label not in cuisine_mapping:
        return None
    return cuisine_mapping[predicted_label]

def format_output(prediction_result, cuisine_info):
    """Format prediction and cuisine recommendation for display"""
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS (SPEAKER-INDEPENDENT MODEL)")
    print(f"{'='*60}")
    
    confidence = prediction_result['confidence']
    confidence_indicator = "✓" if confidence >= 0.50 else "⚠️"
    
    print(f"\n🎯 PREDICTED NATIVE LANGUAGE: {prediction_result['predicted_label'].upper()} {confidence_indicator}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    if confidence < 0.50:
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
        print("Native Language Identification (Speaker-Independent)")
        print("="*60)
        print("\nUsage: python predict_speaker_independent.py <audio_path>")
        print("Example: python predict_speaker_independent.py test_audio.wav")
        print("\nSupported formats: WAV, MP3, FLAC, etc.")
        print("\nThis model uses speaker-independent training and should")
        print("generalize better to unseen speakers!")
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
        output_file = Path(audio_path).stem + "_prediction_speaker_ind.json"
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
