"""
Debug audio loading to see if there's a difference
"""
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"
SAMPLE_RATE = 16000

print("="*70)
print("AUDIO LOADING COMPARISON")
print("="*70)

# Method 1: soundfile (used in extract_hubert_features.py)
print("\n1️⃣ USING SOUNDFILE:")
audio_data, sr = sf.read(audio_path)
waveform1 = torch.FloatTensor(audio_data).t()
if waveform1.dim() == 1:
    waveform1 = waveform1.unsqueeze(0)

# Convert to mono if stereo
if waveform1.shape[0] > 1:
    waveform1 = waveform1.mean(dim=0, keepdim=True)

# Resample
if sr != SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    waveform1 = resampler(waveform1)

print(f"   Shape after processing: {waveform1.shape}")
print(f"   Mean: {waveform1.mean():.10f}")
print(f"   Std: {waveform1.std():.10f}")
print(f"   Min: {waveform1.min():.10f}")
print(f"   Max: {waveform1.max():.10f}")
print(f"   First 10 values: {waveform1.squeeze()[:10]}")

# Method 2: torchaudio (maybe used in batch?)
print("\n2️⃣ USING TORCHAUDIO:")
try:
    waveform2, sr2 = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform2.shape[0] > 1:
        waveform2 = waveform2.mean(dim=0, keepdim=True)
    
    # Resample
    if sr2 != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr2, SAMPLE_RATE)
        waveform2 = resampler(waveform2)
    
    print(f"   Shape after processing: {waveform2.shape}")
    print(f"   Mean: {waveform2.mean():.10f}")
    print(f"   Std: {waveform2.std():.10f}")
    print(f"   Min: {waveform2.min():.10f}")
    print(f"   Max: {waveform2.max():.10f}")
    print(f"   First 10 values: {waveform2.squeeze()[:10]}")
    
    # Compare
    print("\n🔍 COMPARISON:")
    if torch.allclose(waveform1, waveform2, rtol=1e-5):
        print("   ✅ IDENTICAL audio loading!")
    else:
        diff = (waveform1 - waveform2).abs().mean()
        print(f"   ❌ DIFFERENT! Mean difference: {diff:.10f}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
