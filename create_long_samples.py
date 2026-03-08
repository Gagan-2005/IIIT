"""
Create 8-15 second audio samples for testing by concatenating short clips.
"""
import torch
from pathlib import Path
import soundfile as sf
import numpy as np

def get_duration(audio_path):
    """Get audio duration in seconds."""
    info = sf.info(str(audio_path))
    return info.duration

def create_long_sample(source_files, output_path, target_duration=10):
    """Concatenate audio files to create a sample of target duration."""
    waveforms = []
    sample_rate = None
    
    for f in source_files:
        waveform, sr = sf.read(str(f))
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Skip files with different sample rates for simplicity
            continue
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)
            
        waveforms.append(waveform)
        
        # Check if we have enough duration
        total_samples = sum(len(w) for w in waveforms)
        if total_samples / sample_rate >= target_duration:
            break
    
    # Concatenate
    combined = np.concatenate(waveforms)
    
    # Trim to exact target duration
    target_samples = int(target_duration * sample_rate)
    if len(combined) > target_samples:
        combined = combined[:target_samples]
    
    # Save
    sf.write(str(output_path), combined, sample_rate)
    actual_duration = len(combined) / sample_rate
    print(f"Created: {output_path.name} ({actual_duration:.2f}s)")
    return actual_duration

# Process each state
states = ['andhra_pradesh', 'tamil', 'kerala', 'jharkhand', 'karnataka', 'gujrat']

for state in states:
    state_dir = Path(f'data/raw/{state}')
    if not state_dir.exists():
        continue
    
    # Get all wav files
    files = sorted(state_dir.glob('*.wav'))
    if not files:
        continue
    
    print(f"\n{state.upper()}:")
    print(f"Found {len(files)} files")
    
    # Show durations of first few files
    print("Sample durations:")
    for i, f in enumerate(files[:5]):
        dur = get_duration(f)
        print(f"  {f.name}: {dur:.2f}s")
    
    # Create 3 long samples of different durations
    output_dir = Path('test_samples_long')
    output_dir.mkdir(exist_ok=True)
    
    for duration in [8, 10, 15]:
        output_path = output_dir / f"{state}_long_{duration}s.wav"
        try:
            create_long_sample(files, output_path, target_duration=duration)
        except Exception as e:
            print(f"  Error creating {duration}s sample: {e}")

print(f"\n✓ Long samples saved to: test_samples_long/")
print("Upload these via the Gradio UI to test unseen speaker performance.")
