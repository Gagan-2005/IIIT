"""Persistent HuBERT model loader and live feature extraction.

This module loads the HuBERT feature extractor and model ONCE and reuses them
for all subsequent live extraction calls to minimize representation drift
observed when instantiating a fresh model per request.

Provides:
    init_hubert(device=None) -> None   # idempotent
    extract_live_features(audio_path: str) -> np.ndarray (layers x 768)
    get_device() -> str

If calibration parameters (models/live_calibration.npz) exist they can be
applied externally after selecting a layer.
"""

from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel

MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
MAX_DURATION = 30

_feature_extractor = None
_hubert_model = None
_device = None

def init_hubert(device: str | None = None):
    """Load HuBERT model once (idempotent)."""
    global _feature_extractor, _hubert_model, _device
    if _hubert_model is not None:
        return
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _device = device
    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    _hubert_model = HubertModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
    _hubert_model.eval()

def get_device():
    return _device if _device else ('cuda' if torch.cuda.is_available() else 'cpu')

def _load_audio(audio_path: str) -> torch.Tensor:
    audio_data, sr = sf.read(audio_path)
    waveform = torch.FloatTensor(audio_data).t()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:  # mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    max_samples = SAMPLE_RATE * MAX_DURATION
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Note: VAD trimming disabled to avoid preprocessing drift
    # Min duration enforcement also removed - tiling distorts genuine speech patterns
    # Instead, rely on model's confidence gating to flag too-short clips as "unknown"
    return waveform

def extract_live_features(audio_path: str) -> np.ndarray:
    """Return pooled features for all layers: shape (13, 768)."""
    init_hubert()  # ensure loaded
    waveform = _load_audio(audio_path)
    with torch.no_grad():
        inputs = _feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        outputs = _hubert_model(**inputs)
        hidden_states = outputs.hidden_states
        pooled = []
        for layer in hidden_states:
            pooled.append(layer.mean(dim=1).squeeze().cpu().numpy())
        return np.vstack(pooled)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/persistent_hubert.py <audio_path>")
        raise SystemExit(1)
    feats = extract_live_features(sys.argv[1])
    print("Extracted:", feats.shape)