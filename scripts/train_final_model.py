"""
Final Model Training Script with HuBERT Features
Trains the best model using optimal layer (Layer 3) for Native Language Identification
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration
MANIFEST = "metadata.csv"
HUBERT_DIR = "features/hubert"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
BEST_LAYER = 3  # From layer analysis

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 300
USE_SMOTE = True
USE_PCA = True
PCA_COMPONENTS = 128

print("="*80)
print("TRAINING FINAL NLI MODEL WITH HUBERT FEATURES")
print("="*80)

# Load manifest
print(f"\n[1/8] Loading manifest: {MANIFEST}")
df = pd.read_csv(MANIFEST)
print(f"  - Total samples: {len(df)}")

# Detect columns
def detect_column(candidates, df_cols):
    for c in candidates:
        if c in df_cols:
            return c
    return None

utt_col = detect_column(["utt_id", "utt", "id", "filename", "wav_path"], df.columns)
lab_col = detect_column(["label", "lang", "language"], df.columns)

print(f"  - Using columns: utt={utt_col}, label={lab_col}")

# Helper function
def get_utt_from_row(row):
    if utt_col and pd.notna(row.get(utt_col)):
        val = str(row.get(utt_col))
        return Path(val).stem if '/' in val or '\\' in val else val
    for c in ["wav_path", "path", "file"]:
        if c in df.columns and pd.notna(row.get(c)):
            return Path(str(row.get(c))).stem
    return str(row.name)

# Index HuBERT features
print(f"\n[2/8] Indexing HuBERT features from: {HUBERT_DIR}")
hubdir = Path(HUBERT_DIR)
hubert_index = {}
for p in hubdir.rglob("*.npz"):
    hubert_index[p.stem] = p
print(f"  - Indexed {len(hubert_index)} .npz files")

# Extract features from best layer
print(f"\n[3/8] Extracting features from Layer {BEST_LAYER}")
X_list = []
y_list = []
missing = 0

for idx, row in df.iterrows():
    utt = get_utt_from_row(row)
    lab = row[lab_col]
    
    p = hubert_index.get(utt)
    if p is None:
        p = hubert_index.get(utt.replace(" ", "_")) or hubert_index.get(utt.replace("_", " "))
    
    if p is None:
        missing += 1
        continue
    
    try:
        data = np.load(p)
        
        # Extract pooled representation
        if "pooled" in data:
            pooled = data["pooled"]
        elif "hidden_states" in data:
            hs = data["hidden_states"]
            pooled = np.array([h.mean(axis=0) for h in hs])
        else:
            arrays = [data[k] for k in data.files]
            pooled = None
            for arr in arrays:
                if arr.ndim == 2:
                    pooled = arr
                    break
        
        if pooled is not None and pooled.ndim == 2 and pooled.shape[0] > BEST_LAYER:
            X_list.append(pooled[BEST_LAYER])
            y_list.append(lab)
        else:
            missing += 1
    except Exception as e:
        print(f"  Warning: Failed to load {p.name}: {e}")
        missing += 1

X = np.vstack(X_list)
y = np.array(y_list)

print(f"  - Collected {len(X)} samples")
print(f"  - Missing: {missing}")
print(f"  - Feature shape: {X.shape}")

# Label distribution
label_dist = Counter(y)
print(f"  - Label distribution: {dict(label_dist)}")

# Encode labels
print("\n[4/8] Encoding labels")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  - Classes: {list(le.classes_)}")

# Save label encoder
le_path = MODELS_DIR / "label_encoder.joblib"
joblib.dump(le, le_path)
print(f"  - Saved label encoder to: {le_path}")

# Train-test split (stratified)
print(f"\n[5/8] Splitting data (stratified {TEST_SIZE*100:.0f}% test)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"  - Train: {len(X_train)} samples")
print(f"  - Test: {len(X_test)} samples")

# Standardization
print("\n[6/8] Standardizing features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler_path = MODELS_DIR / "scaler_hubert.joblib"
joblib.dump(scaler, scaler_path)
print(f"  - Saved scaler to: {scaler_path}")

# Apply SMOTE for class balancing
if USE_SMOTE:
    print("\n[7/8] Applying SMOTE for class balancing")
    print(f"  - Before SMOTE: {Counter(y_train)}")
    
    # Check minimum class size to determine if SMOTE is feasible
    min_class_size = min(Counter(y_train).values())
    
    if min_class_size >= 6:
        # Use SMOTE with default k_neighbors=5
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  - After SMOTE: {Counter(y_train_balanced)}")
    elif min_class_size >= 2:
        # Use SMOTE with reduced k_neighbors
        k = min_class_size - 1
        print(f"  - Using k_neighbors={k} due to small class sizes")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"  - After SMOTE: {Counter(y_train_balanced)}")
    else:
        # Skip SMOTE if any class has only 1 sample
        print(f"  - Skipping SMOTE: minimum class size is {min_class_size}")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
else:
    X_train_balanced = X_train_scaled
    y_train_balanced = y_train

# PCA for dimensionality reduction
if USE_PCA:
    print(f"\n  - Applying PCA: {X_train_balanced.shape[1]} -> {PCA_COMPONENTS}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_balanced)
    X_test_pca = pca.transform(X_test_scaled)
    pca_path = MODELS_DIR / "pca_hubert.joblib"
    joblib.dump(pca, pca_path)
    print(f"  - Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  - Saved PCA to: {pca_path}")
else:
    X_train_pca = X_train_balanced
    X_test_pca = X_test_scaled

# Train Random Forest Classifier
print("\n[8/8] Training Random Forest Classifier")
print(f"  - Estimators: {N_ESTIMATORS}")
print("  - Training...")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt'
)

clf.fit(X_train_pca, y_train_balanced)
print("  - Training complete!")

# Cross-validation on training set
print("\n  - Running 5-fold cross-validation...")
cv_scores = cross_val_score(clf, X_train_pca, y_train_balanced, cv=5, scoring='accuracy')
print(f"  - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------------------------------------------------------------------------
# Open-set statistics (Mahalanobis distance) for unknown detection
# We compute statistics on the ORIGINAL (pre-SMOTE) training distribution to
# avoid synthetic sample influence. Use PCA space for compact representation.
# ---------------------------------------------------------------------------
print("\n[OPEN-SET] Computing class statistics for unknown detection")
original_train_mask = np.isin(y_encoded, np.unique(y_train))  # original indices in train split
# Reconstruct original train set (pre-SMOTE) in PCA space
X_train_original_scaled = X_train_scaled  # already scaled subset
if USE_PCA:
    X_train_original_pca = pca.transform(X_train_original_scaled)
else:
    X_train_original_pca = X_train_original_scaled

class_means = {}
class_covs = {}
class_inv_covs = {}
class_distances = []  # collect per-sample distance to its true class

for class_idx in np.unique(y_train):
    class_mask = (y_train == class_idx)
    Xc = X_train_original_pca[class_mask]
    mu = Xc.mean(axis=0)
    # Regularized covariance for numerical stability
    centered = Xc - mu
    cov = np.cov(centered, rowvar=False) + np.eye(centered.shape[1]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    class_means[int(class_idx)] = mu
    class_covs[int(class_idx)] = cov
    class_inv_covs[int(class_idx)] = inv_cov
    # Distances for threshold calibration
    for row in centered:
        d = float(row @ inv_cov @ row.T)
        class_distances.append(d)

class_distances = np.array(class_distances)
if len(class_distances) > 0:
    perc95 = float(np.percentile(class_distances, 95))
    distance_threshold = perc95 * 1.15  # safety margin
else:
    distance_threshold = 9999.0
print(f"  - Distance threshold (95th * 1.15): {distance_threshold:.4f}")

# Save statistics
stats_path = MODELS_DIR / "class_stats.npz"
np.savez_compressed(
    stats_path,
    means=class_means,
    covs=class_covs,
    inv_covs=class_inv_covs,
    distance_threshold=distance_threshold,
    pca_components=(PCA_COMPONENTS if USE_PCA else None)
)
print(f"  - Saved class stats to: {stats_path}")

# Predictions
y_train_pred = clf.predict(X_train_pca)
y_test_pred = clf.predict(X_test_pca)

# Evaluation metrics
train_acc = accuracy_score(y_train_balanced, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"\n{'='*80}")
print("MODEL PERFORMANCE")
print(f"{'='*80}")
print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Test F1-Macro:       {test_f1_macro:.4f}")
print(f"Test F1-Weighted:    {test_f1_weighted:.4f}")
print(f"Test Precision:      {test_precision:.4f}")
print(f"Test Recall:         {test_recall:.4f}")

# Classification report
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT")
print(f"{'='*80}")
# Get unique classes in test set
unique_test_classes = np.unique(y_test)
target_names_test = [le.classes_[i] for i in unique_test_classes]
print(classification_report(y_test, y_test_pred, labels=unique_test_classes, target_names=target_names_test))

# Confusion matrix
print(f"\n{'='*80}")
print("CONFUSION MATRIX")
print(f"{'='*80}")
cm = confusion_matrix(y_test, y_test_pred, labels=unique_test_classes)
cm_df = pd.DataFrame(cm, index=target_names_test, columns=target_names_test)
print(cm_df)

# Feature importance
feature_importance = clf.feature_importances_
top_k = min(10, len(feature_importance))
top_indices = np.argsort(feature_importance)[-top_k:][::-1]
print(f"\n{'='*80}")
print(f"TOP {top_k} FEATURE IMPORTANCES")
print(f"{'='*80}")
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. Feature {idx}: {feature_importance[idx]:.6f}")

# Save the trained model
model_path = MODELS_DIR / "rf_hubert_final.joblib"
joblib.dump(clf, model_path)
print(f"\n{'='*80}")
print("MODEL SAVED")
print(f"{'='*80}")
print(f"  - Model: {model_path}")
print(f"  - Scaler: {scaler_path}")
if USE_PCA:
    print(f"  - PCA: {pca_path}")
print(f"  - Label Encoder: {le_path}")

# Save training info
training_info = {
    'best_layer': BEST_LAYER,
    'n_samples': len(X),
    'n_features': X.shape[1],
    'n_classes': len(le.classes_),
    'classes': list(le.classes_),
    'open_set_distance_threshold': distance_threshold,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'use_smote': USE_SMOTE,
    'use_pca': USE_PCA,
    'pca_components': PCA_COMPONENTS if USE_PCA else None,
    'n_estimators': N_ESTIMATORS,
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'label_distribution': {k: int(v) for k, v in label_dist.items()},
}

info_path = MODELS_DIR / "hubert_training_info.json"
with open(info_path, 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"  - Training info: {info_path}")

# Save predictions for analysis
predictions_df = pd.DataFrame({
    'true_label': le.inverse_transform(y_test),
    'predicted_label': le.inverse_transform(y_test_pred),
    'correct': y_test == y_test_pred
})
pred_path = RESULTS_DIR / "rf_final_predictions.csv"
predictions_df.to_csv(pred_path, index=False)
print(f"  - Predictions: {pred_path}")

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}\n")
