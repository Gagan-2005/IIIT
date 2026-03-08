"""Evaluate confusion matrix over cached HuBERT features and report class-wise
accuracy plus most confused pairs. Uses existing Layer 3 pooled features.

Usage:
    python scripts/evaluate_confusion.py [--sample-per-class N]

If --sample-per-class provided, subsamples each class to N examples for speed.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import joblib

HUBERT_DIR = Path("features/hubert")
METADATA = Path("metadata_existing.csv") if Path("metadata_existing.csv").exists() else Path("metadata.csv")
BEST_LAYER = 3

def load_label_encoder():
    le_path = Path("models/label_encoder.joblib")
    if not le_path.exists():
        raise FileNotFoundError("Label encoder not found. Train model first.")
    return joblib.load(le_path)

def collect_samples(subsample: int | None):
    df = pd.read_csv(METADATA)
    # assume columns: wav_path, region?, label? pick last as label if duplicates
    if 'label' in df.columns:
        lab_col = 'label'
    elif 'region' in df.columns:
        lab_col = 'region'
    else:
        lab_col = df.columns[1]
    records = []
    grouped = df.groupby(lab_col)
    for lab, group in grouped:
        if subsample:
            group = group.sample(min(subsample, len(group)), random_state=42)
        for _, row in group.iterrows():
            wav_path = row.get('wav_path', row.iloc[0])
            stem = Path(str(wav_path)).stem
            npz_path = HUBERT_DIR / f"{stem}.npz"
            if npz_path.exists():
                records.append((npz_path, lab))
    return records

def load_layer3(npz_path: Path):
    data = np.load(npz_path)
    pooled = data['pooled']
    return pooled[BEST_LAYER]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-per-class', type=int, default=None, help='Subsample size per class for speed')
    args = ap.parse_args()
    le = load_label_encoder()
    samples = collect_samples(args.sample_per_class)
    if not samples:
        print("No samples collected. Ensure cached features exist.")
        return 1
    model_path = Path("models/rf_hubert_final.joblib")
    scaler_path = Path("models/scaler_hubert.joblib")
    pca_path = Path("models/pca_hubert.joblib")
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path) if pca_path.exists() else None
    X = []
    y_true = []
    for npz_path, lab in samples:
        try:
            vec = load_layer3(npz_path)
            X.append(vec)
            y_true.append(lab)
        except Exception as e:
            print("Skip", npz_path.name, e)
    X = np.vstack(X)
    y_true = le.transform(y_true)
    X_scaled = scaler.transform(X)
    if pca:
        X_proc = pca.transform(X_scaled)
    else:
        X_proc = X_scaled
    y_pred = clf.predict(X_proc)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df)
    # per-class accuracy
    per_class_acc = {}
    for idx, cls in enumerate(le.classes_):
        true_total = cm[idx].sum()
        correct = cm[idx, idx]
        per_class_acc[cls] = correct / true_total if true_total else 0.0
    print("\nPer-class accuracy:")
    for cls, acc in sorted(per_class_acc.items(), key=lambda x: x[1]):
        print(f"  {cls:15s} {acc*100:6.2f}%")
    # most confused pairs
    print("\nTop confused pairs:")
    pairs = []
    for i, cls_i in enumerate(le.classes_):
        for j, cls_j in enumerate(le.classes_):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], cls_i, cls_j))
    for count, a, b in sorted(pairs, reverse=True)[:10]:
        print(f"  {a} -> {b}: {count}")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
