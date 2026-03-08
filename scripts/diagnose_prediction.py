"""
Diagnostic script to understand prediction issues
"""
import sys
from pathlib import Path
import numpy as np
import joblib
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf

# Add scripts directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

print("="*70)
print("🔍 PREDICTION DIAGNOSTICS")
print("="*70)

# Load model and components
print("\n📦 Loading model components...")
model_path = project_root / "models" / "rf_hubert_final.joblib"
scaler_path = project_root / "models" / "scaler_hubert.joblib"
pca_path = project_root / "models" / "pca_hubert.joblib"
encoder_path = project_root / "models" / "label_encoder.joblib"

clf = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
label_encoder = joblib.load(encoder_path)

print(f"   ✓ Classes: {label_encoder.classes_}")
print(f"   ✓ Model: {clf.__class__.__name__}")
print(f"   ✓ Estimators: {clf.n_estimators}")

# Test on multiple Andhra Pradesh samples
print("\n🎯 Testing on Andhra Pradesh samples...")
andhra_dir = project_root / "data" / "raw" / "andhra_pradesh"
andhra_files = list(andhra_dir.glob("*.wav"))[:5]  # Test first 5

if not andhra_files:
    print("   ❌ No Andhra Pradesh audio files found!")
    sys.exit(1)

# Load HuBERT
print("\n📥 Loading HuBERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
hubert_model.eval()

results = []
for audio_file in andhra_files:
    print(f"\n{'='*70}")
    print(f"📄 File: {audio_file.name}")
    
    # Load audio
    audio_array, sr = sf.read(str(audio_file))
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    # Extract features
    with torch.no_grad():
        inputs = feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        ).to(device)
        outputs = hubert_model(**inputs, output_hidden_states=True)
        
        # Get Layer 3
        layer_3 = outputs.hidden_states[3]
        features = layer_3.mean(dim=1).squeeze().cpu().numpy()
    
    print(f"   Feature shape: {features.shape}")
    print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   Feature mean: {features.mean():.4f}")
    print(f"   Feature std: {features.std():.4f}")
    
    # Standardize and PCA
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_pca = pca.transform(features_scaled)
    
    print(f"   After scaling: mean={features_scaled.mean():.4f}, std={features_scaled.std():.4f}")
    print(f"   After PCA: shape={features_pca.shape}")
    
    # Get prediction probabilities
    proba = clf.predict_proba(features_pca)[0]
    pred_idx = np.argmax(proba)
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    
    print(f"\n   🎯 Prediction: {pred_label.upper()}")
    print(f"   Confidence: {proba[pred_idx]*100:.2f}%")
    print(f"\n   📊 All probabilities:")
    for idx, prob in enumerate(proba):
        label = label_encoder.inverse_transform([idx])[0]
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"      {label:15s} {bar} {prob*100:5.2f}%")
    
    # Check feature importance contribution
    feature_importance = clf.feature_importances_[:10]
    print(f"\n   🔍 Top 10 feature values (post-PCA):")
    for i in range(min(10, len(features_pca[0]))):
        print(f"      Feature {i}: {features_pca[0][i]:8.4f} (importance: {feature_importance[i]:.4f})")
    
    results.append({
        'file': audio_file.name,
        'predicted': pred_label,
        'confidence': proba[pred_idx],
        'correct': pred_label == 'andhra_pradesh'
    })

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)
correct = sum(1 for r in results if r['correct'])
print(f"   Correct predictions: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
print(f"   Average confidence: {np.mean([r['confidence'] for r in results])*100:.2f}%")

if correct < len(results):
    print("\n   ⚠️  ISSUE DETECTED: Model is misclassifying Andhra Pradesh samples!")
    print("\n   💡 Possible causes:")
    print("      1. Data quality issues in training set")
    print("      2. Class imbalance still present")
    print("      3. Feature extraction inconsistency")
    print("      4. Need more training samples")
    print("\n   🔧 Recommended fixes:")
    print("      1. Check audio quality and duration")
    print("      2. Use data augmentation")
    print("      3. Try different HuBERT layers")
    print("      4. Increase training samples per class")
else:
    print("\n   ✅ All predictions correct!")
