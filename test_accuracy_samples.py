"""Quick accuracy test on sample files from each region"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from predict import load_models, predict_native_language

# Test samples (2 per region)
test_samples = [
    ("data/raw/andhra_pradesh/Andhra_speaker (1086).wav", "andhra_pradesh"),
    ("data/raw/andhra_pradesh/Andhra_speaker (1230).wav", "andhra_pradesh"),
    ("data/raw/gujrat/Gujrat_speaker_02_2_9.wav", "gujrat"),
    ("data/raw/gujrat/Gujrat_speaker_01_8_4.wav", "gujrat"),
    ("data/raw/jharkhand/Jharkhand_speaker_01_Recording (3)_26.wav", "jharkhand"),
    ("data/raw/jharkhand/Jharkhand_speaker_03_5_25.wav", "jharkhand"),
    ("data/raw/karnataka/Karnataka_speaker_03_1 (511).wav", "karnataka"),
    ("data/raw/karnataka/Karnataka_speaker_03_1 (332).wav", "karnataka"),
    ("data/raw/kerala/Kerala_speaker_05_List65_Splitted_1.wav", "kerala"),
    ("data/raw/kerala/Kerala_speaker_04_List42_Splitted_4.wav", "kerala"),
    ("data/raw/tamil/Tamil_speaker (791).wav", "tamil"),
    ("data/raw/tamil/Tamil_speaker (1316).wav", "tamil"),
]

print("="*70)
print("TESTING PREDICTION ACCURACY ON SAMPLE FILES")
print("="*70)

model, scaler, pca, le = load_models()

correct = 0
total = 0

for audio_path, true_label in test_samples:
    if not Path(audio_path).exists():
        print(f"⚠️  Missing: {Path(audio_path).name}")
        continue
    
    result = predict_native_language(audio_path, model, scaler, pca, le, device='cpu')
    pred_label = result['predicted_label']
    confidence = result['confidence']
    
    is_correct = pred_label == true_label
    correct += is_correct
    total += 1
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {Path(audio_path).stem[:30]:30s} | True: {true_label:15s} | Pred: {pred_label:15s} | Conf: {confidence*100:5.1f}%")

print("="*70)
print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
print("="*70)
