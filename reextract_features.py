"""
Re-extract all HuBERT features using current live extraction pipeline.
This ensures training and inference use identical feature extraction.
"""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "scripts")
from persistent_hubert import extract_live_features, init_hubert


def main():
    print("=== Re-extracting HuBERT Features ===")
    print("This will overwrite features/hubert/*.npz with fresh extractions")
    print("using the current live extraction pipeline.\n")
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Initialize HuBERT once
    init_hubert(device="cpu")
    
    # Find all audio files
    states = ["andhra_pradesh", "tamil", "kerala", "jharkhand", "karnataka", "gujrat"]
    audio_files = []
    
    for state in states:
        state_dir = Path("data/raw") / state
        if state_dir.exists():
            audio_files.extend(state_dir.glob("*.wav"))
    
    print(f"Found {len(audio_files)} audio files\n")
    
    output_dir = Path("features/hubert")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    errors = []
    
    for audio_path in tqdm(audio_files, desc="Extracting"):
        try:
            # Extract using current pipeline
            pooled = extract_live_features(str(audio_path))
            
            # Save in same format as before
            stem = audio_path.stem
            output_path = output_dir / f"{stem}.npz"
            np.savez_compressed(output_path, pooled=pooled)
            
        except Exception as e:
            errors.append((audio_path.name, str(e)))
    
    print(f"\n✓ Extraction complete!")
    print(f"  - Successful: {len(audio_files) - len(errors)}")
    print(f"  - Errors: {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
    
    print("\nNext steps:")
    print("  1. Run: python scripts/train_speaker_normalized.py")
    print("  2. Test: python infer_robust.py 'data/raw/andhra_pradesh/Andhra_speaker (1084).wav'")
    print("  3. Launch UI: python app.py")


if __name__ == "__main__":
    main()
