"""Generate calibration mapping between cached (batch) and live (persistent) HuBERT features.

Creates models/live_calibration.npz containing:
    mean_cached, std_cached, mean_live, std_live for Layer 3 pooled features.

Live adjustment formula:
    live_calibrated = ((live - mean_live) / std_live) * std_cached + mean_cached

This approximates cached distribution so downstream scaler/PCA trained on cached
features sees consistent input.
"""

import numpy as np
from pathlib import Path
import random
from persistent_hubert import extract_live_features

FEATURE_DIR = Path("features/hubert")
OUT_PATH = Path("models/live_calibration.npz")
LAYER = 3  # BEST_LAYER
SAMPLE_COUNT = 60  # number of files to sample across classes

def main():
    npz_files = list(FEATURE_DIR.glob("*.npz"))
    if len(npz_files) == 0:
        print("No cached features found in features/hubert")
        return 1
    random.shuffle(npz_files)
    sample_files = npz_files[:SAMPLE_COUNT]
    cached_vecs = []
    live_vecs = []
    for f in sample_files:
        try:
            data = np.load(f)
            cached = data['pooled'][LAYER]
            cached_vecs.append(cached)
            # derive original wav path by guessing name from metadata pattern
            wav_name = f.stem + '.wav'
            # search in raw subfolders
            candidates = list(Path('data/raw').rglob(wav_name))
            if not candidates:
                continue
            live = extract_live_features(str(candidates[0]))[LAYER]
            live_vecs.append(live)
        except Exception as e:
            print("Skip", f.name, e)
            continue
    if len(live_vecs) < 10:
        print("Insufficient live pairs collected", len(live_vecs))
        return 1
    cached_arr = np.vstack(cached_vecs)
    live_arr = np.vstack(live_vecs)
    mean_cached = cached_arr.mean(axis=0)
    std_cached = cached_arr.std(axis=0) + 1e-8
    mean_live = live_arr.mean(axis=0)
    std_live = live_arr.std(axis=0) + 1e-8
    np.savez_compressed(OUT_PATH, mean_cached=mean_cached, std_cached=std_cached, mean_live=mean_live, std_live=std_live)
    print("Calibration saved:", OUT_PATH)
    print("Cached mean/std (first 5):", mean_cached[:5], std_cached[:5])
    print("Live mean/std   (first 5):", mean_live[:5], std_live[:5])
    diff_mean = np.abs(mean_cached - mean_live).mean()
    print(f"Mean difference before calibration (avg abs): {diff_mean:.6f}")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())