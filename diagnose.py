import argparse
import os
from pathlib import Path

import soundfile as sf

from scripts.predict_backend import predict_from_path


def inspect_audio(path: Path):
    try:
        info = sf.info(str(path))
        print("=== Audio Info ===")
        print(f"Path   : {path}")
        print(f"Format : {info.format}\nSampler: {info.samplerate} Hz\nChannels: {info.channels}\nDuration: {info.duration:.2f} s")
        if info.duration < 5.0:
            print("Note: Very short clip. Try 8–15 seconds for stable results.")
    except Exception as e:
        print(f"Audio read failed: {e}")


def run_diagnosis(path: Path):
    print("\n=== Prediction ===")
    res = predict_from_path(str(path))
    label = res.get("predicted_label")
    conf = res.get("confidence", 0.0)
    top3 = res.get("top_3", [])

    print(f"Label     : {label}")
    print(f"Confidence: {conf:.2%}")
    if top3:
        print("Top-3     :")
        for i, (lbl, c) in enumerate(top3[:3], 1):
            print(f"  {i}. {lbl:16s} {c:.2%}")

    if label == "unknown/uncertain" or conf < 0.55:
        print("\nLow-confidence prediction. Suggestions:")
        print("- Use a longer clip (8–15 seconds) of continuous speech")
        print("- Reduce background noise and keep microphone steady")
        print("- Avoid silence-only or music segments")


def main():
    ap = argparse.ArgumentParser(description="Diagnose prediction issues for a single audio file")
    ap.add_argument("path", nargs="?", help="Path to audio file (.wav)")
    args = ap.parse_args()

    if not args.path:
        print("Provide a path to a .wav file, e.g.\n  python diagnose.py data/raw/andhra_pradesh/Andhra_speaker (1084).wav")
        return

    p = Path(args.path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    inspect_audio(p)
    run_diagnosis(p)


if __name__ == "__main__":
    main()
