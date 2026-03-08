import json
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.predict_backend import predict_from_path


def collect_files() -> List[Path]:
    # Prefer long test samples if present
    long_dir = Path("test_samples_long")
    if long_dir.exists():
        files = sorted(long_dir.glob("*.wav"))
        if files:
            return files

    # Fallback: pick a few from each state
    states = [
        "andhra_pradesh",
        "tamil",
        "kerala",
        "jharkhand",
        "karnataka",
        "gujrat",
    ]
    files: List[Path] = []
    for state in states:
        d = Path("data/raw") / state
        if not d.exists():
            continue
        files.extend(sorted(d.glob("*.wav"))[:3])
    return files


def run_batch(files: List[Path]) -> Dict[str, Dict[str, float]]:
    """Run predictions and aggregate counts + average confidences per predicted label."""
    counts: Dict[str, int] = {}
    conf_sums: Dict[str, float] = {}

    for f in files:
        r = predict_from_path(str(f))
        label = r.get("predicted_label", "unknown/uncertain")
        conf = float(r.get("confidence", 0.0))
        counts[label] = counts.get(label, 0) + 1
        conf_sums[label] = conf_sums.get(label, 0.0) + conf
        print(f"{f.name:40s} -> {label:18s} {conf:.2%}")

    # Build summary
    summary: Dict[str, Dict[str, float]] = {}
    for label, n in counts.items():
        avg_conf = conf_sums[label] / max(n, 1)
        summary[label] = {"count": float(n), "avg_conf": avg_conf}
    return summary


def main():
    files = collect_files()
    if not files:
        print("No test audio files found.")
        return

    print(f"Testing {len(files)} files...\n")
    summary = run_batch(files)

    print("\n=== Summary ===")
    for label, stats in summary.items():
        print(f"{label:18s} count={int(stats['count']):3d}  avg_conf={stats['avg_conf']:.2%}")


if __name__ == "__main__":
    main()
