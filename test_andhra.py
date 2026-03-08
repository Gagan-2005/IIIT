"""Test Andhra predictions"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from predict_backend import predict_from_path

# Get Andhra samples
andhra_dir = Path("data/raw/andhra_pradesh")
andhra_files = list(andhra_dir.glob("*.wav"))[:5]

print("="*70)
print("TESTING ANDHRA PRADESH SAMPLES")
print("="*70)

for audio_file in andhra_files:
    result = predict_from_path(str(audio_file), device='cpu')
    predicted = result['predicted_label']
    confidence = result['confidence']
    
    status = "✓" if predicted == "andhra_pradesh" else "✗"
    
    print(f"\n{status} {audio_file.name}")
    print(f"  Predicted: {predicted} ({confidence*100:.2f}%)")
    
    if predicted != "andhra_pradesh":
        top3 = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top 3:")
        for label, prob in top3:
            print(f"    - {label}: {prob*100:.2f}%")

print("\n" + "="*70)
