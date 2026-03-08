"""
Quick test script - Run this before launching UI
This verifies the backend predictions are correct
"""
import sys
sys.path.insert(0, 'scripts')
from predict_backend import predict_from_path
from pathlib import Path

print("\n" + "="*70)
print("PRE-LAUNCH VERIFICATION")
print("="*70)

# Test a few samples from each region
test_samples = {
    "Andhra": "data/raw/andhra_pradesh/Andhra_speaker (1084).wav",
    "Tamil": "data/raw/tamil/Tamil_speaker (1).wav",
    "Karnataka": "data/raw/karnataka/Karnataka_speaker_03_1 (1).wav",
}

all_correct = True

for region, filepath in test_samples.items():
    if not Path(filepath).exists():
        print(f"\n❌ {region}: File not found: {filepath}")
        continue
        
    result = predict_from_path(filepath, device='cpu')
    pred = result['predicted_label']
    conf = result['confidence']
    
    # Map region names to expected labels
    expected_map = {
        "andhra": "andhra_pradesh",
        "tamil": "tamil",
        "karnataka": "karnataka",
        "gujarat": "gujrat",
        "kerala": "kerala",
        "jharkhand": "jharkhand"
    }
    expected = expected_map.get(region.lower(), region.lower().replace(' ', '_'))
    is_correct = pred == expected
    
    if is_correct:
        print(f"\n✓ {region}: {pred} ({conf*100:.1f}%)")
    else:
        print(f"\n✗ {region}: WRONG! Predicted {pred} ({conf*100:.1f}%)")
        all_correct = False

print("\n" + "="*70)
if all_correct:
    print("✅ ALL TESTS PASSED - UI should work correctly!")
    print("\nLaunch UI with: python app.py")
    print("Then test with your Andhra file in the browser.")
else:
    print("❌ SOME TESTS FAILED - There's still an issue!")
print("="*70 + "\n")
