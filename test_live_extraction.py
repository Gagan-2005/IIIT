from scripts.persistent_hubert import init_hubert, extract_live_features
from pathlib import Path
import numpy as np

# Test live extraction
init_hubert()

# Find a test audio file
audio_files = list(Path('data/raw').rglob('*.wav'))
if audio_files:
    test_audio = str(audio_files[0])
    print(f"Testing with: {test_audio}")
    
    # Extract live
    live_features = extract_live_features(test_audio)
    print(f"\nLive extraction shape: {live_features.shape}")
    
    # Compare with cached
    cached_file = Path('features/hubert') / f"{Path(test_audio).stem}.npz"
    if cached_file.exists():
        cached_data = np.load(cached_file)
        print(f"Cached shape: {cached_data['pooled'].shape}")
        
        if live_features.shape != cached_data['pooled'].shape:
            print("\n⚠️ SHAPE MISMATCH!")
            print(f"  Live: {live_features.shape}")
            print(f"  Cached: {cached_data['pooled'].shape}")
            print(f"  This explains why predictions are wrong!")
    else:
        print("No cached version found")
else:
    print("No audio files found in data/raw")
