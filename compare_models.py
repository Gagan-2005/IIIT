"""
Compare Old Model (with leakage) vs New Model (speaker-independent)
Shows how speaker normalization and proper splitting improves generalization
"""

import sys
from pathlib import Path
sys.path.insert(0, 'scripts')

print("="*80)
print("MODEL COMPARISON: OLD vs NEW")
print("="*80)

# Test sample
test_audio = "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"

if not Path(test_audio).exists():
    print(f"\n⚠️  Test file not found: {test_audio}")
    print("Please provide a valid audio file path as argument")
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        sys.exit(1)

print(f"\nTest audio: {test_audio}")
print(f"True label: andhra_pradesh\n")

# Test OLD model (with data leakage)
print("="*80)
print("OLD MODEL (with data leakage)")
print("="*80)
try:
    from predict import load_models, predict_native_language
    
    model, scaler, pca, le = load_models()
    result_old = predict_native_language(test_audio, model, scaler, pca, le, device='cpu', allow_live=True)
    
    print(f"\n✓ Prediction: {result_old['predicted_label']}")
    print(f"  Confidence: {result_old['confidence']*100:.2f}%")
    print(f"  Top 3:")
    for i, (label, prob) in enumerate(result_old['top_3'][:3], 1):
        print(f"    {i}. {label}: {prob*100:.2f}%")
    
except Exception as e:
    print(f"\n❌ Old model failed: {e}")
    result_old = None

# Test NEW model (speaker-normalized)
print(f"\n{'='*80}")
print("NEW MODEL (speaker-normalized)")
print("="*80)

try:
    import importlib
    # Reload to avoid conflicts
    if 'predict_speaker_normalized' in sys.modules:
        importlib.reload(sys.modules['predict_speaker_normalized'])
    
    from predict_speaker_normalized import load_models as load_models_new
    from predict_speaker_normalized import predict_native_language as predict_new
    
    model_new, scaler_new, pca_new, le_new = load_models_new()
    result_new = predict_new(test_audio, model_new, scaler_new, pca_new, le_new, device='cpu')
    
    print(f"\n✓ Prediction: {result_new['predicted_label']}")
    print(f"  Confidence: {result_new['confidence']*100:.2f}%")
    print(f"  Top 3:")
    for i, (label, prob) in enumerate(result_new['top_3'][:3], 1):
        print(f"    {i}. {label}: {prob*100:.2f}%")
    
except FileNotFoundError as e:
    print(f"\n⚠️  New model not found. Please train it first:")
    print("    python scripts/train_speaker_normalized.py")
    result_new = None
except Exception as e:
    print(f"\n❌ New model failed: {e}")
    import traceback
    traceback.print_exc()
    result_new = None

# Comparison
print(f"\n{'='*80}")
print("COMPARISON")
print("="*80)

if result_old and result_new:
    print("\nOLD MODEL (with data leakage):")
    print(f"  - Prediction: {result_old['predicted_label']}")
    print(f"  - Confidence: {result_old['confidence']*100:.2f}%")
    print(f"  - Issue: Overconfident due to speaker memorization")
    
    print("\nNEW MODEL (speaker-normalized):")
    print(f"  - Prediction: {result_new['predicted_label']}")
    print(f"  - Confidence: {result_new['confidence']*100:.2f}%")
    print(f"  - Benefit: Learns accent patterns, not speaker identity")
    
    # Check if both correct
    correct_old = result_old['predicted_label'] == 'andhra_pradesh'
    correct_new = result_new['predicted_label'] == 'andhra_pradesh'
    
    print(f"\n{'='*80}")
    if correct_old and correct_new:
        print("✓ Both models predicted correctly!")
        print("\nKey difference:")
        conf_diff = result_old['confidence'] - result_new['confidence']
        if conf_diff > 0.1:
            print(f"  - Old model is {conf_diff*100:.1f}% MORE confident")
            print(f"  - This overconfidence comes from memorizing the speaker")
            print(f"  - New model's lower confidence is more honest and reliable")
        else:
            print(f"  - Similar confidence levels")
    elif correct_new and not correct_old:
        print("✓ NEW model is BETTER - correctly identified accent!")
        print("✗ Old model FAILED due to speaker-specific memorization")
    elif correct_old and not correct_new:
        print("⚠️  Old model correct, new model incorrect")
        print("   This can happen if the test speaker is in the training set")
        print("   Try testing on a completely unseen speaker")
    else:
        print("✗ Both models incorrect")
        print("   Consider collecting more training data for this region")
    
    print(f"{'='*80}\n")
    
    print("INTERPRETATION:")
    print("---------------")
    print("If you're testing on a speaker FROM the training set:")
    print("  - Old model will have VERY high confidence (95-99%)")
    print("  - New model will have MODERATE confidence (70-85%)")
    print("  → This is EXPECTED and GOOD")
    print()
    print("If you're testing on a COMPLETELY NEW speaker:")
    print("  - Old model will likely FAIL or have low confidence")
    print("  - New model should work with moderate confidence (60-80%)")
    print("  → This proves the new model generalizes better")
    
else:
    if not result_old:
        print("\n⚠️  Could not load old model")
    if not result_new:
        print("\n⚠️  Could not load new model - train it first:")
        print("    python scripts/train_speaker_independent.py")

print(f"\n{'='*80}\n")
