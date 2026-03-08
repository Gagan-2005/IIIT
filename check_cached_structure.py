"""Check cached feature structure"""
import numpy as np
from pathlib import Path

cached_path = Path("features/hubert/Tamil_speaker (1).npz")
data = np.load(cached_path, allow_pickle=True)

print("Cached file structure:")
print(f"  Keys: {list(data.keys())}")

if 'pooled' in data:
    pooled = data['pooled']
    print(f"\n  'pooled' shape: {pooled.shape}")
    print(f"  'pooled' ndim: {pooled.ndim}")
    
    if pooled.ndim == 2:
        print(f"\n  Interpretation:")
        print(f"    - Option 1: (n_layers=26, n_features=768) - layers dimension")
        print(f"    - Option 2: (n_frames, n_features=768) - time frames dimension")
        
        # Check which makes sense
        print(f"\n  pooled[0] shape: {pooled[0].shape}")
        print(f"  pooled[0] first 5 values: {pooled[0][:5]}")
        print(f"\n  pooled[3] shape: {pooled[3].shape}")
        print(f"  pooled[3] first 5 values: {pooled[3][:5]}")
        
        # Try mean
        mean_pooled = pooled.mean(axis=0)
        print(f"\n  mean(axis=0) shape: {mean_pooled.shape}")
        print(f"  mean(axis=0) first 5 values: {mean_pooled[:5]}")
