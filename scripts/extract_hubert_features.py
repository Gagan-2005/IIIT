"""
HuBERT Feature Extraction for Single Audio File
Extracts layer-wise HuBERT features for inference
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds

# Set deterministic behavior
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

def extract_hubert_features(audio_path, output_path=None, device='cpu'):
    """
    Extract HuBERT features from an audio file
    
    Args:
        audio_path: Path to audio file (wav/mp3/etc)
        output_path: Path to save .npz file (optional)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with 'pooled' key containing layer-wise features
    """
    print(f"Loading audio: {audio_path}")
    
    # Load audio - use soundfile directly to avoid torchaudio backend issues
    try:
        import soundfile as sf
        # Load with soundfile directly
        audio_data, sr = sf.read(audio_path)
        # Convert to torch tensor and ensure correct shape (channels, samples)
        waveform = torch.FloatTensor(audio_data).t()
        # Ensure 2D tensor (add channel dimension if needed)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}\nMake sure soundfile is installed: pip install soundfile")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        print(f"  - Resampling from {sr}Hz to {SAMPLE_RATE}Hz")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Trim to max duration
    max_samples = MAX_DURATION * SAMPLE_RATE
    if waveform.shape[1] > max_samples:
        print(f"  - Trimming to {MAX_DURATION}s")
        waveform = waveform[:, :max_samples]
    
    print(f"  - Audio shape: {waveform.shape}")
    print(f"  - Duration: {waveform.shape[1]/SAMPLE_RATE:.2f}s")
    
    # Load HuBERT model
    print(f"Loading HuBERT model: {MODEL_NAME}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model = model.to(device)
    model.eval()
    
    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        # Prepare input
        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # Tuple of tensors (one per layer)
        
        # Pool each layer (mean pooling over time dimension)
        pooled_features = []
        for layer_idx, layer_output in enumerate(hidden_states):
            # layer_output: (batch, time, features)
            pooled = layer_output.mean(dim=1).squeeze().cpu().numpy()
            pooled_features.append(pooled)
            print(f"  - Layer {layer_idx}: {pooled.shape}")
        
        pooled_features = np.vstack(pooled_features)  # (n_layers, feature_dim)
    
    print(f"Extracted features shape: {pooled_features.shape}")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pooled=pooled_features)
        print(f"Saved features to: {output_path}")
    
    return {'pooled': pooled_features}

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_hubert_features.py <audio_path> [output_path]")
        print("Example: python extract_hubert_features.py audio.wav features/hubert/audio.npz")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Auto-generate output path if not provided
    if output_path is None:
        audio_name = Path(audio_path).stem
        output_path = f"features/hubert/{audio_name}.npz"
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Extract features
    try:
        features = extract_hubert_features(audio_path, output_path, device)
        print("\nSuccess!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
