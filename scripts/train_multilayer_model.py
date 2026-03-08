"""Train a multilayer HuBERT RandomForest using concatenated pooled vectors
from all 13 layers, followed by StandardScaler + PCA + SMOTE balancing.

Produces artifacts:
    models/rf_hubert_multilayer.joblib
    models/scaler_multilayer.joblib
    models/pca_multilayer.joblib (if PCA enabled)
    models/label_encoder_multilayer.joblib
    models/multilayer_training_info.json

Usage:
    python scripts/train_multilayer_model.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

METADATA = Path("metadata_existing.csv") if Path("metadata_existing.csv").exists() else Path("metadata.csv")
HUBERT_DIR = Path("features/hubert")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 400
USE_PCA = True
PCA_COMPONENTS = 512
USE_SMOTE = True

def pick_label_column(df):
    for c in ["label", "language", "region"]:
        if c in df.columns:
            return c
    return df.columns[-1]

def stem_from_row(row):
    for c in ["wav_path", "path", "file"]:
        if c in row and pd.notna(row[c]):
            return Path(str(row[c])).stem
    return Path(str(row.get('wav_path', ''))).stem

def load_samples(df, lab_col):
    index = {p.stem: p for p in HUBERT_DIR.rglob("*.npz")}
    X_list, y_list = [] , []
    missing = 0
    expected_dim = None
    for _, row in df.iterrows():
        stem = stem_from_row(row)
        lab = row[lab_col]
        p = index.get(stem) or index.get(stem.replace(" ", "_")) or index.get(stem.replace("_", " "))
        if not p:
            missing += 1
            continue
        try:
            data = np.load(p)
            if 'pooled' in data:
                pooled = data['pooled']
            else:
                arrays = [data[k] for k in data.files]
                pooled = None
                for arr in arrays:
                    if arr.ndim == 2 and arr.shape[1] == 768:
                        pooled = arr
                        break
            if pooled is None:
                missing += 1
                continue
            vec = pooled.reshape(-1)
            if expected_dim is None:
                expected_dim = vec.shape[0]
            if vec.shape[0] != expected_dim:
                # Skip inconsistent dimension (possibly different layer count)
                missing += 1
                continue
            X_list.append(vec)
            y_list.append(lab)
        except Exception:
            missing += 1
    if not X_list:
        return np.empty((0,0)), np.array([]), missing
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, missing

def main():
    df = pd.read_csv(METADATA)
    lab_col = pick_label_column(df)
    X, y, missing = load_samples(df, lab_col)
    if len(X) == 0:
        print("No samples loaded. Ensure cached features exist.")
        return 1
    print(f"Loaded samples: {len(X)} missing: {missing} feature_dim: {X.shape[1]}")
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    if USE_SMOTE:
        cnt = Counter(y_train)
        min_sz = min(cnt.values())
        if min_sz >= 2:
            k = min(5, min_sz - 1) if min_sz > 2 else 1
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
        else:
            X_train_bal, y_train_bal = X_train_s, y_train
    else:
        X_train_bal, y_train_bal = X_train_s, y_train
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X_train_p = pca.fit_transform(X_train_bal); X_test_p = pca.transform(X_test_s)
    else:
        pca = None; X_train_p = X_train_bal; X_test_p = X_test_s
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        max_features="sqrt"
    )
    clf.fit(X_train_p, y_train_bal)
    y_pred = clf.predict(X_test_p)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    print(f"Test Accuracy: {acc:.4f} F1-macro: {f1m:.4f}")
    # save artifacts
    joblib.dump(le, MODELS_DIR / "label_encoder_multilayer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_multilayer.joblib")
    if pca:
        joblib.dump(pca, MODELS_DIR / "pca_multilayer.joblib")
    joblib.dump(clf, MODELS_DIR / "rf_hubert_multilayer.joblib")
    info = {
        'n_samples': len(X),
        'feature_dim_raw': X.shape[1],
        'use_pca': USE_PCA,
        'pca_components': PCA_COMPONENTS if USE_PCA else None,
        'test_accuracy': float(acc),
        'test_f1_macro': float(f1m),
        'classes': list(le.classes_),
        'missing': missing
    }
    import json
    with open(MODELS_DIR / "multilayer_training_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print("Saved multilayer model artifacts.")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
