"""Simulate exact UI prediction flow"""
import sys
sys.path.insert(0, 'scripts')

# This is exactly what the UI does
from predict_backend import predict_from_path

# Test with different Andhra samples
test_files = [
    "data/raw/andhra_pradesh/Andhra_speaker (1084).wav",  # In dataset
    "data/raw/andhra_pradesh/Andhra_speaker (1083).wav",  # NOT in dataset
    "data/raw/tamil/Tamil_speaker (1).wav",  # Tamil control
]

print("="*70)
print("SIMULATING UI PREDICTION FLOW")
print("="*70)

for audio_file in test_files:
    print(f"\n{'='*70}")
    print(f"Testing: {audio_file}")
    print('='*70)
    
    try:
        result = predict_from_path(audio_file, device='cpu')
        
        print(f"\nPredicted: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Unknown flag: {result.get('unknown', False)}")
        
        print(f"\nTop 3:")
        sorted_probs = sorted(result['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        for i, (label, prob) in enumerate(sorted_probs[:3], 1):
            print(f"  {i}. {label}: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("If you see Jharkhand predictions above, there's a problem.")
print("If you see correct predictions, the issue is elsewhere (browser cache?).")
print("="*70)
