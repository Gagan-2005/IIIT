"""
COMPLETE RESET SCRIPT
Run this to ensure everything is fresh
"""
import subprocess
import sys
import time

print("="*70)
print("COMPLETE SYSTEM RESET")
print("="*70)

# 1. Kill all Python processes
print("\n[1/4] Stopping all Python processes...")
try:
    subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                  capture_output=True, text=True)
    print("  ✓ Python processes stopped")
except:
    print("  - No Python processes running")

# 2. Clear Python cache
print("\n[2/4] Clearing Python cache...")
import shutil
import os
from pathlib import Path

cache_dirs = [
    Path("__pycache__"),
    Path("scripts/__pycache__"),
]

for cache_dir in cache_dirs:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"  ✓ Removed {cache_dir}")

# 3. Verify models
print("\n[3/4] Verifying speaker-normalized models...")
model_dir = Path("models/speaker_normalized")
required_files = [
    "rf_hubert.joblib",
    "scaler.joblib",
    "pca.joblib",
    "label_encoder.joblib"
]

all_exist = True
for file in required_files:
    filepath = model_dir / file
    if filepath.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ MISSING: {file}")
        all_exist = False

if not all_exist:
    print("\n❌ ERROR: Missing model files!")
    print("Run: python scripts/train_speaker_normalized.py")
    sys.exit(1)

# 4. Test prediction
print("\n[4/4] Testing prediction...")
sys.path.insert(0, 'scripts')
from predict_backend import predict_from_path

test_file = "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"
result = predict_from_path(test_file, device='cpu')

if result['predicted_label'] == 'andhra_pradesh':
    print(f"  ✓ Andhra test: {result['confidence']*100:.1f}%")
else:
    print(f"  ✗ Andhra test FAILED: predicted {result['predicted_label']}")

print("\n" + "="*70)
print("✅ RESET COMPLETE!")
print("\nNow launch UI:")
print("  python app.py")
print("\nThen open browser in INCOGNITO/PRIVATE mode:")
print("  http://127.0.0.1:7860")
print("="*70)
