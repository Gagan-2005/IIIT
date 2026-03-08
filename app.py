"""
Simple web app to predict Indian state from English speech
Also recommends a dish from that region!
"""

import gradio as gr
import json
import joblib
import numpy as np
from pathlib import Path
import sys

from scripts.extract_hubert_features import extract_hubert_features  # legacy direct extraction (kept for compatibility)
try:
    # Use unified backend for prediction + logging if available
    from scripts.predict_backend import predict_from_path, log_edge_case
    _USE_BACKEND = True
except Exception:
    _USE_BACKEND = False

# Configuration
MODELS_DIR = Path("models")
BEST_LAYER = 3
CUISINE_MAPPING_PATH = "cuisine_mapping.json"

# Global variables for models
model = None
scaler = None
pca = None
le = None
cuisine_mapping = None
device = 'cpu'
pair_clf = None
pair_scaler = None
pair_pca = None
pair_info = None

def load_all_models():
    """Load all required models and data"""
    global model, scaler, pca, le, cuisine_mapping, device, pair_clf, pair_scaler, pair_pca, pair_info
    
    print("Loading models...")
    
    # Check device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load speaker-normalized models
    model_path = MODELS_DIR / "speaker_normalized" / "rf_hubert.joblib"
    scaler_path = MODELS_DIR / "speaker_normalized" / "scaler.joblib"
    pca_path = MODELS_DIR / "speaker_normalized" / "pca.joblib"
    le_path = MODELS_DIR / "speaker_normalized" / "label_encoder.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nPlease run scripts/train_speaker_normalized.py first!")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    if pca_path.exists():
        pca = joblib.load(pca_path)
    
    # Optional: load pairwise Andhra vs Jharkhand verifier artifacts
    try:
        aj_model_path = MODELS_DIR / "andhra_jharkhand_verifier.joblib"
        aj_scaler_path = MODELS_DIR / "andhra_jharkhand_scaler.joblib"
        aj_pca_path = MODELS_DIR / "andhra_jharkhand_pca.joblib"
        aj_info_path = MODELS_DIR / "andhra_jharkhand_info.json"
        if aj_model_path.exists() and aj_scaler_path.exists() and aj_info_path.exists():
            pair_clf = joblib.load(aj_model_path)
            pair_scaler = joblib.load(aj_scaler_path)
            pair_pca = joblib.load(aj_pca_path) if aj_pca_path.exists() else None
            with open(aj_info_path, 'r', encoding='utf-8') as f:
                pair_info = json.load(f)
            print("Loaded Andhra/Jharkhand verifier")
        else:
            print("Pairwise verifier not found (optional)")
    except Exception as e:
        print(f"Warning: Failed to load pairwise verifier: {e}")

    # Load cuisine mapping
    with open(CUISINE_MAPPING_PATH, 'r', encoding='utf-8') as f:
        cuisine_mapping = json.load(f)
    
    print("Models loaded successfully!")

def predict_from_audio(audio_path):
    """Gradio callback: run prediction via backend if available, else legacy fallback."""
    if audio_path is None:
        return "Please upload an audio file."

    # Preferred path: unified backend (handles cached vs live, open-set, logging)
    if _USE_BACKEND:
        try:
            print(f"\n[DEBUG] Using backend prediction for: {audio_path}")
            result = predict_from_path(audio_path)
            # Attempt logging (ignore failures silently)
            try:
                log_edge_case(audio_path, result)
            except Exception:
                pass
            predicted_label = result['predicted_label']
            confidence = result['confidence']
            print(f"[DEBUG] Backend prediction: {predicted_label} ({confidence*100:.2f}%)")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            cuisine_info = cuisine_mapping.get(predicted_label, None)
            return format_results(predicted_label, confidence, sorted_probs, cuisine_info, result)
        except Exception as e:
            import traceback
            return f"Backend prediction error:\n{e}\n{traceback.format_exc()}"

    # Fallback: legacy direct feature extraction (no open-set / logging)
    try:
        print(f"[DEBUG] Using FALLBACK (direct extraction) for: {audio_path}")
        features_dict = extract_hubert_features(audio_path, device=device)
        pooled = features_dict['pooled']
        print(f"[DEBUG] Pooled shape: {pooled.shape}")
        # Average all 26 layers (training does this too)
        layer_vec = pooled.mean(axis=0).reshape(1, -1)
        layer_vec = scaler.transform(layer_vec)
        if pca is not None:
            layer_vec = pca.transform(layer_vec)
        probs = model.predict_proba(layer_vec)[0]
        classes = le.classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        predicted_label = classes[np.argmax(probs)]
        confidence = float(np.max(probs))
        print(f"[DEBUG] Fallback prediction: {predicted_label} ({confidence*100:.2f}%)")
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        cuisine_info = cuisine_mapping.get(predicted_label, None)
        legacy_result = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': prob_dict,
            'unknown': False,
            'mahal_distance': None
        }
        return format_results(predicted_label, confidence, sorted_probs, cuisine_info, legacy_result)
    except Exception as e:
        import traceback
        return f"Legacy fallback error:\n{e}\n{traceback.format_exc()}"

def format_results(predicted_label, confidence, sorted_probs, cuisine_info, full_result):
    """Build markdown output block for UI."""
    lines = []
    lines.append("=" * 70)
    lines.append("🎤 NATIVE LANGUAGE IDENTIFICATION RESULT")
    lines.append("=" * 70)

    display_label = predicted_label.replace('_', ' ').title() if predicted_label else 'Unknown'
    if full_result.get('unknown'):
        lines.append(f"\n**Predicted:** {display_label} (flagged as UNKNOWN / out-of-distribution)")
    else:
        lines.append(f"\n**Predicted:** {display_label}")
    lines.append(f"**Confidence:** {confidence*100:.2f}%")
    if full_result.get('mahal_distance') is not None:
        lines.append(f"**Mahalanobis Distance:** {full_result['mahal_distance']:.3f}")

    # Top 3 predictions
    lines.append("\n### Top 3 Predictions:")
    for i, (label, prob) in enumerate(sorted_probs[:3], 1):
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        lines.append(f"{i}. **{label.replace('_', ' ').title()}**: {bar} {prob*100:.2f}%")

    # Cuisine section
    lines.append("\n" + "=" * 70)
    lines.append("🍽️ CUISINE RECOMMENDATION")
    lines.append("=" * 70)
    if cuisine_info:
        lines.append(f"\n**Region:** {cuisine_info['region']}")
        lines.append(f"**Language:** {cuisine_info['language']}")
        lines.append(f"**Cuisine:** {cuisine_info['cuisine']}")
        lines.append("\n### 📜 Must-Try Dishes:")
        for i, dish in enumerate(cuisine_info['dishes'], 1):
            lines.append(f"{i}. {dish}")
        lines.append("\n### ✨ Characteristics:")
        for char in cuisine_info['characteristics']:
            lines.append(f"• {char}")
    else:
        lines.append("\n⚠️ No cuisine information available for this region.")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)

def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container { font-family: 'Inter', 'Arial', sans-serif; }
    .output-text { font-size: 14px; line-height: 1.6; }
    /* Hide default Gradio logo and footer */
    .logo, .footer, .wrap .logo-wrap, .app .footer { display: none !important; }
    /* Optional custom header styling */
    h1, h2, h3 { font-weight: 600; }
    """
    
    # Interface description
    description = """
    ## 🎤 Native Language Identification & Cuisine Recommender
    
    Upload an audio file (speech recording) to:
    1. **Identify the speaker's native language** using HuBERT-based features and a calibrated classifier
    2. **Get personalized cuisine recommendations** based on the identified region
    
    ### ✅ Works Best
    - Single speaker, natural continuous speech (not stitched)
    - 8–15 seconds, low background noise
    - 16 kHz sample rate, mono (1 channel)
    - Steady mic distance (avoid moving the phone)
    
    ### 📝 Instructions
    - Upload a clear recording (WAV preferred; MP3/FLAC also supported)
    - Speak naturally (counting or reading a paragraph is fine)
    - If the result shows "Unknown/Uncertain", try a longer, cleaner clip
    
    ### 🌏 Supported Regions
    Andhra Pradesh • Gujarat • Jharkhand • Karnataka • Kerala • Tamil Nadu
    """
    
    examples = [
        # You can add example audio files here if available
    ]
    
    # Create interface
    interface = gr.Interface(
        fn=predict_from_audio,
        inputs=gr.Audio(
            type="filepath",
            label="Upload Audio File",
            sources=["upload", "microphone"]
        ),
        outputs=gr.Textbox(
            label="Prediction Results",
            lines=30,
            max_lines=50
        ),
        title="🎯 Native Language Identifier & Cuisine Recommender",
        description=description,
        examples=examples,
        theme=gr.themes.Soft(),
        css=custom_css,
        allow_flagging="never"
    )
    
    return interface

def main():
    """Main function to launch the web app"""
    print("="*70)
    print("Native Language Identification Web Interface")
    print("="*70)
    
    # Load models
    try:
        load_all_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nPlease ensure you have trained the model first by running:")
        print("  python scripts/train_final_model.py")
        sys.exit(1)
    
    # Create and launch interface
    print("\nLaunching web interface...")
    interface = create_interface()
    
    # Launch with share=True to get a public link (optional)
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )

if __name__ == "__main__":
    main()
