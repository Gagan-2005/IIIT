import platform
import sys
from pathlib import Path

REQUIRED_MODEL_FILES = [
    Path("models/speaker_normalized/rf_hubert.joblib"),
    Path("models/speaker_normalized/scaler.joblib"),
    Path("models/speaker_normalized/pca.joblib"),
    Path("models/speaker_normalized/label_encoder.joblib"),
]


def check_python():
    print("=== Python / OS ===")
    print(f"Python : {sys.version.split()[0]}")
    print(f"OS     : {platform.system()} {platform.release()}")
    print(f"Arch   : {platform.machine()}")


def check_packages():
    print("\n=== Packages ===")
    def try_import(name):
        try:
            mod = __import__(name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{name:12s} OK   version={ver}")
            return True
        except Exception as e:
            print(f"{name:12s} FAIL {e}")
            return False

    has_torch = try_import("torch")
    try_import("numpy")
    try_import("transformers")
    try_import("soundfile")
    # torchaudio can be optional for these helpers
    try:
        import torchaudio  # type: ignore
        print(f"{'torchaudio':12s} OK   version={getattr(torchaudio, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{'torchaudio':12s} WARN {e}")

    if has_torch:
        import torch  # type: ignore
        print(f"CUDA   : {'Yes' if torch.cuda.is_available() else 'No'}")
        if torch.cuda.is_available():
            print(f"GPUs   : {torch.cuda.device_count()}")


def check_models():
    print("\n=== Model Artifacts ===")
    missing = []
    for p in REQUIRED_MODEL_FILES:
        exists = p.exists()
        print(f"{str(p):64s} {'OK' if exists else 'MISSING'}")
        if not exists:
            missing.append(p)
    if missing:
        print("\nOne or more model files are missing. Re-run training or restore artifacts.")


def quick_prediction_check():
    """Try a quick prediction on a known cached file if available."""
    try:
        from scripts.predict_backend import predict_from_path
    except Exception as e:
        print(f"\nPredict backend import failed: {e}")
        return

    candidate = Path("data/raw/andhra_pradesh/Andhra_speaker (1084).wav")
    if not candidate.exists():
        print("\nQuick check: sample file not found; skipping prediction test.")
        return

    try:
        r = predict_from_path(str(candidate), device="cpu")
        print("\n=== Quick Prediction Check ===")
        print(f"File      : {candidate.name}")
        print(f"Label     : {r.get('predicted_label')}")
        print(f"Confidence: {r.get('confidence', 0.0):.2%}")
    except Exception as e:
        print(f"\nPrediction failed: {e}")


def main():
    check_python()
    check_packages()
    check_models()
    quick_prediction_check()


if __name__ == "__main__":
    main()
