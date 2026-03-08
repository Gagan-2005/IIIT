import numpy as np
from pathlib import Path

# Load first cached feature file
files = list(Path('features/hubert').glob('*.npz'))
if files:
    data = np.load(files[0])
    print(f"File: {files[0].name}")
    print(f"Keys: {list(data.keys())}")
    print(f"Pooled shape: {data['pooled'].shape}")
    print(f"\nInterpretation:")
    print(f"  Shape {data['pooled'].shape} means:")
    if data['pooled'].shape[0] == 26:
        print(f"    - 26 layers × {data['pooled'].shape[1]} features per layer")
        print(f"    - Axis 0 = layers, Axis 1 = features")
    else:
        print(f"    - {data['pooled'].shape[0]} time frames × {data['pooled'].shape[1]} features")
        print(f"    - Axis 0 = time, Axis 1 = features")
else:
    print("No cached features found!")
