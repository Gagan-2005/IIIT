"""
Quick Demo: Native Language Identification with Cuisine Recommendation
Shows how to use the trained model for predictions
"""

import sys
from pathlib import Path

# Check if models exist
models_dir = Path("models")
required_files = [
    "rf_hubert_final.joblib",
    "scaler_hubert.joblib",
    "pca_hubert.joblib",
    "label_encoder.joblib"
]

print("="*70)
print("NLI + Cuisine Recommender - Quick Demo")
print("="*70)

print("\n✅ Checking model files...")
all_exist = True
for file in required_files:
    path = models_dir / file
    if path.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} - MISSING!")
        all_exist = False

if not all_exist:
    print("\n❌ Some model files are missing!")
    print("Please run: python scripts/train_final_model.py")
    sys.exit(1)

print("\n✅ All model files found!")

# Check if we have sample audio files
print("\n📁 Checking for sample audio files...")
sample_dirs = Path("data/raw").glob("*")
sample_found = False

for dir in sample_dirs:
    if dir.is_dir():
        wav_files = list(dir.glob("*.wav"))[:1]  # Get first wav file
        if wav_files:
            sample_file = wav_files[0]
            sample_found = True
            print(f"  ✓ Found sample: {sample_file}")
            break

print("\n" + "="*70)
print("HOW TO USE")
print("="*70)

print("\n1️⃣  Command-Line Prediction:")
print("   python scripts/predict.py path/to/audio.wav")

if sample_found:
    print(f'\n   Example with your data:')
    print(f'   python scripts/predict.py "{sample_file}"')

print("\n2️⃣  Web Interface (Recommended):")
print("   python app.py")
print("   Then open: http://127.0.0.1:7860")

print("\n3️⃣  Extract Features from New Audio:")
print("   python scripts/extract_hubert_features.py audio.wav")

print("\n" + "="*70)
print("WHAT YOU'LL GET")
print("="*70)

print("""
🎯 Predicted Native Language (with confidence score)
📊 Top 3 predictions with probabilities
🍽️  Regional Cuisine Recommendations:
   • Region & Language info
   • Must-try dishes (7-8 items)
   • Cuisine characteristics
""")

print("="*70)
print("MODEL PERFORMANCE")
print("="*70)
print("""
✓ Test Accuracy: 94.83%
✓ Supported Languages: 6 Indian regions
  - Andhra Pradesh (Telugu)
  - Gujarat (Gujarati)
  - Jharkhand (Hindi/Tribal)
  - Karnataka (Kannada)
  - Kerala (Malayalam)
  - Tamil Nadu (Tamil)
""")

print("="*70)
print("READY TO GO! 🚀")
print("="*70)
print("\nTry one of the commands above to start predicting!\n")
