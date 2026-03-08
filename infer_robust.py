import argparse
import json
from pathlib import Path
from typing import Any, Dict

# Use the unified backend already used by the UI
from scripts.predict_backend import predict_from_path


def main() -> None:
    p = argparse.ArgumentParser(description="Robust inference runner for NLI HuBERT model")
    p.add_argument("path", type=str, help="Path to an audio .wav file")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    p.add_argument("--topk", type=int, default=3, help="Top-K classes to show")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()

    audio_path = Path(args.path)
    if not audio_path.exists():
        raise SystemExit(f"File not found: {audio_path}")

    # Run backend prediction (handles cache-first + live extraction + gating)
    result: Dict[str, Any] = predict_from_path(str(audio_path), device=args.device)

    # Trim top-K for display
    result["top_3"] = result.get("top_3", [])[: args.topk]

    if args.json:
        print(json.dumps(result, indent=2))
        return

    label = result.get("predicted_label")
    conf = result.get("confidence", 0.0)
    top3 = result.get("top_3", [])

    print("\n=== Inference Result ===")
    print(f"File      : {audio_path}")
    print(f"Label     : {label}")
    print(f"Confidence: {conf:.2%}")
    if top3:
        print("Top-K     :")
        for i, (lbl, c) in enumerate(top3, 1):
            print(f"  {i}. {lbl:16s} {c:.2%}")

    # Simple gating message
    if label == "unknown/uncertain" or conf < 0.55:
        print("\nNote: Low confidence prediction. Try a longer, cleaner clip (8–15s).")


if __name__ == "__main__":
    main()
