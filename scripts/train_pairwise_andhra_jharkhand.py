"""Train a pairwise verifier to distinguish Andhra Pradesh vs Jharkhand.

Uses concatenated pooled HuBERT layers (13 x 768) with optional PCA and a
LogisticRegression classifier. Falls back to extracting features if .npz
missing. Produces artifacts:
  models/andhra_jharkhand_verifier.joblib
  models/andhra_jharkhand_scaler.joblib
  models/andhra_jharkhand_pca.joblib (if PCA enabled)
  models/andhra_jharkhand_info.json

Run:
  python scripts/train_pairwise_andhra_jharkhand.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from extract_hubert_features import extract_hubert_features

METADATA = Path("metadata_existing.csv") if Path("metadata_existing.csv").exists() else Path("metadata.csv")
FEATURE_DIR = Path("features/hubert")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_CLASSES = {"andhra_pradesh", "jharkhand"}
TEST_SIZE = 0.25
RANDOM_STATE = 42
USE_PCA = True
PCA_COMPONENTS = 128  # adjust to <= min(n_samples, n_features)

def resolve_label_column(df):
    for c in ["label", "language", "region"]:
        if c in df.columns:
            return c
    return df.columns[1]

def stem_from_row(row):
    for c in ["wav_path", "path", "file"]:
        if c in row and pd.notna(row[c]):
            return Path(str(row[c])).stem
    return Path(str(row.get('wav_path', ''))).stem

def load_or_extract(stem, wav_path):
    npz_path = FEATURE_DIR / f"{stem}.npz"
    pooled = None
    if npz_path.exists():
        data = np.load(npz_path)
        if 'pooled' in data:
            pooled = data['pooled']
        else:
            for k in data.files:
                arr = data[k]
                if arr.ndim == 2 and arr.shape[1] == 768:
                    pooled = arr
                    break
    if pooled is None:
        pooled = extract_hubert_features(wav_path, output_path=str(npz_path))['pooled']
    return pooled

def main():
    if not METADATA.exists():
        print("Metadata file missing")
        return 1
    df = pd.read_csv(METADATA)
    lab_col = resolve_label_column(df)
    subset = df[df[lab_col].isin(TARGET_CLASSES)].copy()
    if subset.empty:
        print("No samples for target classes.")
        return 1
    X_list = []
    y_list = []
    for _, row in subset.iterrows():
        lab = row[lab_col]
        wav_path = row.get('wav_path', None)
        if wav_path is None:
            continue
        wav_path = str(wav_path)
        if not Path(wav_path).exists():
            continue
        stem = stem_from_row(row)
        try:
            pooled = load_or_extract(stem, wav_path)
            # enforce exactly 13 layers (HuBERT base) else skip
            if pooled.shape[0] != 13 or pooled.shape[1] != 768:
                continue
            vec = pooled.reshape(-1)
            X_list.append(vec)
            y_list.append(lab)
        except Exception as e:
            print("Skip", stem, e)
            continue
    if not X_list:
        print("No feature vectors loaded.")
        return 1
    X = np.vstack(X_list)
    y = np.array(y_list)
    # Encode manually: andhra_pradesh -> 0, jharkhand -> 1
    label_to_int = {c: i for i, c in enumerate(sorted(TARGET_CLASSES))}
    y_int = np.array([label_to_int[v] for v in y])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_int
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if USE_PCA:
        max_c = min(PCA_COMPONENTS, X_train_s.shape[0], X_train_s.shape[1])
        pca = PCA(n_components=max_c, random_state=RANDOM_STATE)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
    else:
        pca = None
        X_train_p = X_train_s
        X_test_p = X_test_s
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = {c: w for c, w in zip(classes, class_weights)}
    clf = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight=cw,
        max_iter=1000,
        solver='lbfgs'
    )
    clf.fit(X_train_p, y_train)
    y_pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    print(f"Pairwise Test Accuracy: {acc:.4f} F1-macro: {f1m:.4f}")
    # Save artifacts
    joblib.dump(clf, MODELS_DIR / "andhra_jharkhand_verifier.joblib")
    joblib.dump(scaler, MODELS_DIR / "andhra_jharkhand_scaler.joblib")
    if pca:
        joblib.dump(pca, MODELS_DIR / "andhra_jharkhand_pca.joblib")
    info = {
        'classes': sorted(TARGET_CLASSES),
        'mapping': label_to_int,
        'test_accuracy': float(acc),
        'test_f1_macro': float(f1m),
        'pca_components': PCA_COMPONENTS if USE_PCA else None,
        'n_samples': int(len(X))
    }
    with open(MODELS_DIR / "andhra_jharkhand_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print("Saved pairwise verifier artifacts.")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
