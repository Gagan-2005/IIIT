import numpy as np
from pathlib import Path

# Check multiple cached files
cached_dir = Path('features/hubert')
files = list(cached_dir.glob('*.npz'))[:5]  # Check first 5 files

print("Checking cached feature shapes:")
for f in files:
    data = np.load(f)
    print(f"  {f.name}: {data['pooled'].shape}")

# Check if there's any pattern - 26 = 13 * 2?
print(f"\nObservation: Shape is (26, 768)")
print(f"  - 26 = 13 * 2 (might be concatenating two HuBERT runs?)")
print(f"  - Or using a different model with 24 layers + input/output?")
print(f"  - Or stacking forward + backward pass?")
