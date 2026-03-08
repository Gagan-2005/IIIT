"""
Test Script: Quick prediction test with a sample audio file
"""

import sys
from pathlib import Path

# Find a sample audio file
sample_dir = Path("data/raw")
sample_file = None

for region_dir in sample_dir.glob("*"):
    if region_dir.is_dir():
        wav_files = list(region_dir.glob("*.wav"))
        if wav_files:
            sample_file = wav_files[0]
            break

if not sample_file:
    print("❌ No sample audio files found in data/raw/")
    print("Please ensure you have audio files in the data directory.")
    sys.exit(1)

print("="*70)
print("🧪 TESTING PREDICTION SYSTEM")
print("="*70)
print(f"\n📁 Using sample audio: {sample_file}")
print(f"   Region folder: {sample_file.parent.name}")

# Add scripts to path
sys.path.insert(0, str(Path("scripts")))

# Import prediction functions
try:
    from predict import (
        load_models, 
        predict_native_language, 
        load_cuisine_mapping,
        format_output
    )
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Make sure you're in the project root directory.")
    sys.exit(1)

print("\n📦 Loading models...")
try:
    model, scaler, pca, le = load_models()
    print("   ✓ Models loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    print("\n💡 Solution: Run 'python scripts/train_final_model.py' first")
    sys.exit(1)

print("\n📖 Loading cuisine database...")
try:
    cuisine_mapping = load_cuisine_mapping()
    print("   ✓ Cuisine mapping loaded")
except Exception as e:
    print(f"   ✗ Error loading cuisine mapping: {e}")
    sys.exit(1)

print("\n🎯 Running prediction...")
try:
    # Check device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    print(f"   Using device: {device}")
    
    # Predict
    result = predict_native_language(
        str(sample_file), 
        model, 
        scaler, 
        pca, 
        le, 
        device
    )
    
    # Get cuisine info
    cuisine_info = cuisine_mapping.get(result['predicted_label'])
    
    # Format and display
    format_output(result, cuisine_info)
    
    print("\n✅ TEST SUCCESSFUL!")
    print("\n💡 Next steps:")
    print("   1. Try with your own audio: python scripts/predict.py your_audio.wav")
    print("   2. Launch web interface: python app.py")
    print("   3. Read QUICKSTART.md for more options")
    
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
