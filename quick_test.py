"""Quick accuracy test"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')

from predict import load_models, predict_native_language

# Load models
model, scaler, pca, le = load_models()

# Test files with expected labels
tests = [
    ('data/raw/andhra_pradesh/Andhra_speaker (1083).wav', 'andhra_pradesh'),
    ('data/raw/kerala/Kerala_speaker_04_List42_Splitted_4.wav', 'kerala'),
    ('data/raw/tamil/Tamil_speaker (1).wav', 'tamil'),
    ('data/raw/karnataka/Karnataka_speaker (1).wav', 'karnataka'),
]

print("\n" + "="*70)
print("ACCURACY TEST")
print("="*70)

correct = 0
total = 0

for audio_file, expected_label in tests:
    if not Path(audio_file).exists():
        print(f"SKIP: {Path(audio_file).name} (file not found)")
        continue
    
    result = predict_native_language(audio_file, model, scaler, pca, le)
    predicted = result['predicted_label']
    confidence = result['confidence']
    
    is_correct = (predicted == expected_label)
    correct += is_correct
    total += 1
    
    status = "CORRECT" if is_correct else "WRONG"
    print(f"\n{Path(audio_file).name}:")
    print(f"  Expected: {expected_label}")
    print(f"  Predicted: {predicted} ({confidence*100:.2f}%)")
    print(f"  Status: {status}")

print("\n" + "="*70)
print(f"FINAL ACCURACY: {correct}/{total} = {100*correct/total:.2f}%")
print("="*70)
