"""Check which Andhra samples are in test set vs training"""
import pandas as pd
from pathlib import Path

df = pd.read_csv('metadata_existing.csv')

# Check specific samples
test_samples = [
    "Andhra_speaker (1083).wav",
    "Andhra_speaker (1084).wav",
    "Andhra_speaker (1085).wav"
]

print("Checking if samples are in dataset:\n")
for sample in test_samples:
    full_path = f"data/raw/andhra_pradesh/{sample}"
    exists = full_path in df['wav_path'].values
    print(f"{sample}: {'IN DATASET' if exists else 'NOT IN DATASET'}")

# Load training results to see test set predictions
print("\n" + "="*70)
print("Checking test set predictions from training...")
import json
with open('models/speaker_normalized/training_info.json', 'r') as f:
    info = json.load(f)

print(f"\nTest accuracy for Andhra Pradesh: {info.get('per_class_accuracy', {}).get('andhra_pradesh', 'N/A')}")
print(f"Overall test accuracy: {info['test_accuracy']*100:.2f}%")
