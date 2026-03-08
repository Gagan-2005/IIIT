import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_backend import predict_from_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_predict_backend.py <audio_path>")
        sys.exit(1)
    audio_path = sys.argv[1]
    result = predict_from_path(audio_path)
    print("\n=== Prediction Result ===")
    print(f"File: {audio_path}")
    print(f"Predicted Label: {result['predicted_label']} (base: {result['base_label']})")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Margin: {result['margin']:.4f}")
    print(f"Low Confidence: {result['low_confidence']}")
    print(f"Unknown Flag: {result['unknown']}")
    print("Top 3:")
    for lbl, prob in result['top_3']:
        print(f"  - {lbl}: {prob:.4f}")
    print("All Probabilities (truncated):")
    for k, v in list(result['all_probabilities'].items())[:10]:
        print(f"  {k}: {v:.4f}")
