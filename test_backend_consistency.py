"""Test if predict_backend gives same results as predict_speaker_normalized"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')

from predict_backend import predict_from_path

# Test on a Tamil sample
audio_path = "data/raw/tamil/Tamil_speaker (1).wav"

print("Testing prediction consistency...\n")
print("="*70)
print(f"Testing: {audio_path}")
print("="*70)

result = predict_from_path(audio_path, device='cpu')

print(f"\nPredicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
print(f"\nTop 3:")
all_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
for i, (label, prob) in enumerate(all_probs[:3], 1):
    print(f"  {i}. {label}: {prob*100:.2f}%")

print("\n" + "="*70)

# Test on Gujarat
guj_path = list(Path("data/raw/gujrat").glob("*.wav"))[0]
print(f"\nTesting: {guj_path}")
print("="*70)

result2 = predict_from_path(str(guj_path), device='cpu')

print(f"\nPredicted: {result2['predicted_label']}")
print(f"Confidence: {result2['confidence']*100:.2f}%")
print(f"\nTop 3:")
all_probs2 = sorted(result2['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
for i, (label, prob) in enumerate(all_probs2[:3], 1):
    print(f"  {i}. {label}: {prob*100:.2f}%")
