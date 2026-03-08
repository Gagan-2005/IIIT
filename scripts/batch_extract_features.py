"""
Batch Feature Extraction Script
Extracts HuBERT features for all audio files in the dataset
"""

import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import torch

# Add scripts to path
sys.path.insert(0, "scripts")
from extract_hubert_features import extract_hubert_features

# Configuration
METADATA = "metadata_existing.csv"  # Use filtered metadata with only existing files
HUBERT_DIR = Path("features/hubert")
HUBERT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("BATCH HUBERT FEATURE EXTRACTION")
print("="*70)

# Load metadata
print(f"\nLoading metadata: {METADATA}")
df = pd.read_csv(METADATA)
print(f"Total audio files: {len(df)}")

# Check existing features
existing_features = set(p.stem for p in HUBERT_DIR.glob("*.npz"))
print(f"Already extracted: {len(existing_features)}")
print(f"Remaining: {len(df) - len(existing_features)}")

# Filter to only process files that don't have features yet
def get_feature_name(row):
    wav_path = str(row['wav_path'])
    return Path(wav_path).stem

df['feature_name'] = df.apply(get_feature_name, axis=1)
df_remaining = df[~df['feature_name'].isin(existing_features)]

if len(df_remaining) == 0:
    print("\n✅ All features already extracted!")
    sys.exit(0)

print(f"\nProcessing {len(df_remaining)} files...")
print("This may take several hours depending on your hardware.")
print("Press Ctrl+C to stop at any time.\n")

# Check device
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    device = 'cpu'

print(f"Using device: {device}\n")

# Load HuBERT model once (major speed improvement!)
print("Loading HuBERT model (one-time)...")
try:
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2FeatureExtractor, HubertModel
    
    MODEL_NAME = "facebook/hubert-base-ls960"
    SAMPLE_RATE = 16000
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    hubert_model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    hubert_model = hubert_model.to(device)
    hubert_model.eval()
    print("✓ Model loaded successfully!\n")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    print("Falling back to per-file model loading (slower)...")
    hubert_model = None

# Process each file
success_count = 0
error_count = 0
errors_log = []

for idx, row in tqdm(df_remaining.iterrows(), total=len(df_remaining), desc="Extracting"):
    audio_path = row['wav_path']
    feature_name = row['feature_name']
    output_path = HUBERT_DIR / f"{feature_name}.npz"
    
    # Skip if output already exists
    if output_path.exists():
        success_count += 1
        continue
    
    # Check if audio file exists
    if not Path(audio_path).exists():
        error_count += 1
        errors_log.append(f"File not found: {audio_path}")
        continue
    
    try:
        if hubert_model is not None:
            # Fast path: use pre-loaded model
            audio_data, sr = sf.read(audio_path)
            waveform = torch.FloatTensor(audio_data).t()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # Extract features
            with torch.no_grad():
                inputs = feature_extractor(
                    waveform.squeeze().numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = hubert_model(**inputs)
                hidden_states = outputs.hidden_states
                
                pooled_features = []
                for layer_output in hidden_states:
                    pooled = layer_output.mean(dim=1).squeeze().cpu().numpy()
                    pooled_features.append(pooled)
                
                pooled_features = np.vstack(pooled_features)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, pooled=pooled_features)
            success_count += 1
        else:
            # Slow path: load model each time
            extract_hubert_features(audio_path, output_path, device)
            success_count += 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user!")
        break
    except Exception as e:
        error_count += 1
        errors_log.append(f"Error processing {audio_path}: {str(e)}")
        continue

print(f"\n{'='*70}")
print("EXTRACTION COMPLETE")
print(f"{'='*70}")
print(f"✅ Successfully extracted: {success_count}")
print(f"❌ Errors: {error_count}")

if errors_log:
    print(f"\nError log:")
    for error in errors_log[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(errors_log) > 10:
        print(f"  ... and {len(errors_log) - 10} more errors")

print(f"\nFeatures saved to: {HUBERT_DIR}")
print("\nNext steps:")
print("1. Run: python scripts/train_final_model.py")
print("2. Test: python test_prediction.py")
