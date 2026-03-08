"""
Debug: Check audio shape after loading with soundfile
"""
import soundfile as sf
import torch
import numpy as np

audio_path = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"

print("="*70)
print("AUDIO LOADING DEBUG")
print("="*70)

# Load with soundfile
audio_data, sr = sf.read(audio_path)
print(f"\nRaw soundfile output:")
print(f"  audio_data type: {type(audio_data)}")
print(f"  audio_data shape: {audio_data.shape}")
print(f"  audio_data dtype: {audio_data.dtype}")
print(f"  Sample rate: {sr}")
print(f"  First 5 samples: {audio_data[:5]}")

# Current approach (with transpose)
waveform_with_transpose = torch.FloatTensor(audio_data).t()
print(f"\nWith .t() (transpose):")
print(f"  Shape: {waveform_with_transpose.shape}")
if waveform_with_transpose.dim() == 1:
    waveform_with_transpose = waveform_with_transpose.unsqueeze(0)
    print(f"  After unsqueeze(0): {waveform_with_transpose.shape}")

# Without transpose
waveform_no_transpose = torch.FloatTensor(audio_data)
print(f"\nWithout .t():")
print(f"  Shape: {waveform_no_transpose.shape}")
if waveform_no_transpose.dim() == 1:
    waveform_no_transpose = waveform_no_transpose.unsqueeze(0)
    print(f"  After unsqueeze(0): {waveform_no_transpose.shape}")

# Check if they're the same
if torch.equal(waveform_with_transpose, waveform_no_transpose):
    print(f"\n✅ SAME waveform!")
else:
    print(f"\n❌ DIFFERENT waveforms!")
    print(f"  With transpose mean: {waveform_with_transpose.mean():.10f}")
    print(f"  Without transpose mean: {waveform_no_transpose.mean():.10f}")

print("\n" + "="*70)
