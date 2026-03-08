"""Simple live prediction test without cached HuBERT features.

Ensures the pipeline performs persistent live extraction + calibration and
predicts the expected label for a known sample. Run after removing any
corresponding cached feature .npz files.

Usage:
    python scripts/test_live_prediction.py

Exit code 0 if prediction matches expected label; 1 otherwise.
"""

from pathlib import Path
import sys
from predict import load_models, predict_native_language

# Known sample (expected native language label from metadata)
AUDIO_PATH = "data/raw/andhra_pradesh/Andhra_speaker (1083).wav"
EXPECTED_LABEL = "andhra_pradesh"


def main():
    if not Path(AUDIO_PATH).exists():
        print(f"Missing audio sample: {AUDIO_PATH}")
        return 1

    try:
        model, scaler, pca, le = load_models()
        result = predict_native_language(
            AUDIO_PATH, model, scaler, pca, le, allow_live=True
        )
        pred = result["predicted_label"]
        conf = result["confidence"]
        print("Predicted:", pred, f"(confidence {conf*100:.2f}%)")
        if pred == EXPECTED_LABEL:
            print("SUCCESS: Live prediction matches expected label.")
            return 0
        else:
            print("FAIL: Expected", EXPECTED_LABEL, "but got", pred)
            # Show top 3 for debugging
            print("Top 3:", result["top_3"])
            return 1
    except Exception as e:
        print("Error during live prediction test:", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
