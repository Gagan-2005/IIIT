"""Debug: Test exact prediction with detailed output"""
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')

# Replicate exactly what happens
from predict_backend import predict_from_path
import numpy as np

# Test Andhra file
audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"

print("="*70)
print("DETAILED PREDICTION DEBUG")
print("="*70)
print(f"\nTesting: {audio_path}")

# Check if file exists
if not Path(audio_path).exists():
    print(f"ERROR: File not found!")
    sys.exit(1)

print(f"File exists: ✓")

# Check cached features
from pathlib import Path
audio_name = Path(audio_path).stem
cached_path = Path("features/hubert") / f"{audio_name}.npz"

if cached_path.exists():
    print(f"Cached features found: {cached_path}")
    data = np.load(cached_path)
    pooled = data['pooled']
    print(f"  - Pooled shape: {pooled.shape}")
    print(f"  - Layer 0 first 3 values: {pooled[0, :3]}")
    print(f"  - Layer 3 first 3 values: {pooled[3, :3]}")
    
    # Show what averaging does
    avg = pooled.mean(axis=0)
    print(f"  - Averaged first 3 values: {avg[:3]}")
else:
    print("No cached features - will use live extraction")

# Now run actual prediction
print("\n" + "="*70)
print("RUNNING PREDICTION")
print("="*70)

result = predict_from_path(audio_path, device='cpu')

print(f"\n✓ Predicted: {result['predicted_label']}")
print(f"  Confidence: {result['confidence']*100:.2f}%")
print(f"  Unknown flag: {result.get('unknown', False)}")

print(f"\n  Top 5:")
sorted_probs = sorted(result['all_probabilities'].items(), 
                     key=lambda x: x[1], reverse=True)
for i, (label, prob) in enumerate(sorted_probs[:5], 1):
    print(f"    {i}. {label}: {prob*100:.2f}%")

print("\n" + "="*70)

# If prediction is wrong, investigate
if result['predicted_label'] != 'andhra_pradesh':
    print("❌ WRONG PREDICTION!")
    print("\nInvestigating cause...")
    
    # Check model files
    from pathlib import Path
    model_dir = Path("models/speaker_normalized")
    print(f"\nModel directory: {model_dir}")
    print(f"  Exists: {model_dir.exists()}")
    if model_dir.exists():
        files = list(model_dir.glob("*.joblib"))
        print(f"  Files: {[f.name for f in files]}")
else:
    print("✓ CORRECT PREDICTION!")

print("="*70)
