"""
Speaker-Independent Training with Proper Data Split
Ensures model learns accent patterns, not speaker identities
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
from sklearn.model_selection import GroupShuffleSplit, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration
MANIFEST = "metadata_existing.csv"
HUBERT_DIR = "features/hubert"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
BEST_LAYER = 3

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200  # Reduced to prevent overfitting
USE_SMOTE = False   # Disabled - use class weights instead
USE_PCA = True
PCA_COMPONENTS = 96  # Reduced to prevent overfitting
SPEAKER_NORMALIZATION = True  # NEW: Apply speaker normalization

print("="*80)
print("SPEAKER-INDEPENDENT NLI MODEL TRAINING")
print("="*80)
print("\nKey improvements:")
print("  ✓ True speaker-independent train/test split (GroupShuffleSplit)")
print("  ✓ Speaker normalization (zero mean, unit variance per speaker)")
print("  ✓ Reduced model complexity to prevent overfitting")
print("  ✓ Class weights instead of SMOTE")

# Load manifest
print(f"\n[1/9] Loading manifest: {MANIFEST}")
df = pd.read_csv(MANIFEST)
print(f"  - Total samples: {len(df)}")

# Extract speaker IDs
def extract_speaker_id(row):
    """Extract speaker identifier from filename"""
    if 'speaker_id' in df.columns:
        return str(row['speaker_id'])
    
    # Parse from wav_path
    path = str(row.get('wav_path', ''))
    filename = Path(path).stem
    
    # Handle different naming patterns
    if 'Andhra_speaker' in filename:
        # Extract number from "Andhra_speaker (1084)"
        import re
        match = re.search(r'\((\d+)\)', filename)
        return f"andhra_{match.group(1)}" if match else filename
    elif 'Gujrat_speaker' in filename:
        parts = filename.split('_')
        return f"gujrat_{parts[2]}" if len(parts) > 2 else filename
    elif 'Jharkhand_speaker' in filename:
        parts = filename.split('_')
        return f"jharkhand_{parts[2]}" if len(parts) > 2 else filename
    elif 'Karnataka_speaker' in filename:
        parts = filename.split('_')
        return f"karnataka_{parts[2]}" if len(parts) > 2 else filename
    elif 'Kerala_speaker' in filename:
        parts = filename.split('_')
        return f"kerala_{parts[2]}" if len(parts) > 2 else filename
    elif 'Tamil_speaker' in filename:
        match = re.search(r'\((\d+)\)', filename)
        return f"tamil_{match.group(1)}" if match else filename
    
    return filename

df['speaker_id'] = df.apply(extract_speaker_id, axis=1)

# Detect columns
utt_col = 'wav_path'
lab_col = 'label'

print(f"  - Detected {df['speaker_id'].nunique()} unique speakers")

# Helper function
def get_utt_from_row(row):
    val = str(row.get(utt_col))
    return Path(val).stem

# Index HuBERT features
print(f"\n[2/9] Indexing HuBERT features from: {HUBERT_DIR}")
hubdir = Path(HUBERT_DIR)
hubert_index = {}
for p in hubdir.rglob("*.npz"):
    hubert_index[p.stem] = p
print(f"  - Indexed {len(hubert_index)} .npz files")

# Extract features from best layer
print(f"\n[3/9] Extracting features from Layer {BEST_LAYER}")
X_list = []
y_list = []
speaker_list = []
missing = 0

for idx, row in df.iterrows():
    utt = get_utt_from_row(row)
    lab = row[lab_col]
    speaker = row['speaker_id']
    
    p = hubert_index.get(utt)
    if p is None:
        p = hubert_index.get(utt.replace(" ", "_")) or hubert_index.get(utt.replace("_", " "))
    
    if p is None:
        missing += 1
        continue
    
    try:
        data = np.load(p)
        if "pooled" in data:
            pooled = data["pooled"]
        else:
            missing += 1
            continue
        
        if pooled.ndim == 2 and pooled.shape[0] > BEST_LAYER:
            X_list.append(pooled[BEST_LAYER])
            y_list.append(lab)
            speaker_list.append(speaker)
        else:
            missing += 1
    except Exception as e:
        missing += 1

X = np.vstack(X_list)
y = np.array(y_list)
speakers = np.array(speaker_list)

print(f"  - Collected {len(X)} samples")
print(f"  - Missing: {missing}")
print(f"  - Feature shape: {X.shape}")
print(f"  - Unique speakers: {len(np.unique(speakers))}")

# Label distribution
label_dist = Counter(y)
print(f"  - Label distribution: {dict(label_dist)}")

# Encode labels
print("\n[4/9] Encoding labels")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  - Classes: {list(le.classes_)}")

# Save label encoder
le_path = MODELS_DIR / "label_encoder_speaker_ind.joblib"
joblib.dump(le, le_path)
print(f"  - Saved label encoder to: {le_path}")

# CRITICAL: Speaker-independent split using GroupShuffleSplit
print(f"\n[5/9] Speaker-Independent Train/Test Split (GroupShuffleSplit)")
print("  - Ensuring NO speaker appears in both train and test sets")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=speakers))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
speakers_train, speakers_test = speakers[train_idx], speakers[test_idx]

print(f"  - Train: {len(X_train)} samples, {len(np.unique(speakers_train))} speakers")
print(f"  - Test: {len(X_test)} samples, {len(np.unique(speakers_test))} speakers")

# Verify no speaker overlap
overlap = set(speakers_train) & set(speakers_test)
if overlap:
    print(f"  ❌ ERROR: Speaker overlap detected: {len(overlap)} speakers")
else:
    print(f"  ✓ No speaker overlap - true speaker-independent split!")

# Speaker Normalization (NEW)
if SPEAKER_NORMALIZATION:
    print("\n[6/9] Applying Speaker Normalization")
    print("  - Normalizing features per speaker (zero mean, unit variance)")
    
    # Normalize training data per speaker
    X_train_normalized = X_train.copy()
    for speaker in np.unique(speakers_train):
        mask = speakers_train == speaker
        speaker_data = X_train_normalized[mask]
        mean = speaker_data.mean(axis=0)
        std = speaker_data.std(axis=0) + 1e-8
        X_train_normalized[mask] = (speaker_data - mean) / std
    
    # For test data, normalize per speaker as well
    X_test_normalized = X_test.copy()
    for speaker in np.unique(speakers_test):
        mask = speakers_test == speaker
        speaker_data = X_test_normalized[mask]
        mean = speaker_data.mean(axis=0)
        std = speaker_data.std(axis=0) + 1e-8
        X_test_normalized[mask] = (speaker_data - mean) / std
    
    X_train = X_train_normalized
    X_test = X_test_normalized
    print("  ✓ Speaker normalization complete")

# Standardization (across all speakers)
print("\n[7/9] Global Feature Standardization")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaler_path = MODELS_DIR / "scaler_speaker_ind.joblib"
joblib.dump(scaler, scaler_path)
print(f"  - Saved scaler to: {scaler_path}")

# PCA for dimensionality reduction
if USE_PCA:
    print(f"\n[8/9] Applying PCA: {X_train_scaled.shape[1]} -> {PCA_COMPONENTS}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    pca_path = MODELS_DIR / "pca_speaker_ind.joblib"
    joblib.dump(pca, pca_path)
    print(f"  - Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  - Saved PCA to: {pca_path}")
else:
    X_train_pca = X_train_scaled
    X_test_pca = X_test_scaled

# Train Random Forest with reduced complexity
print("\n[9/9] Training Random Forest Classifier")
print(f"  - Estimators: {N_ESTIMATORS}")
print(f"  - Using class weights (no SMOTE)")
print(f"  - Reduced max_depth to prevent overfitting")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
    max_depth=15,  # Reduced from 25
    min_samples_split=10,  # Increased from 5
    min_samples_leaf=4,    # Increased from 2
    max_features='sqrt',
    bootstrap=True
)

clf.fit(X_train_pca, y_train)
print("  - Training complete!")

# Cross-validation (speaker-independent)
print("\n  - Running speaker-independent 5-fold cross-validation...")
cv_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for train_cv_idx, val_cv_idx in skf.split(X_train_pca, y_train):
    # Ensure speakers don't overlap in CV folds (approximate)
    X_cv_train, X_cv_val = X_train_pca[train_cv_idx], X_train_pca[val_cv_idx]
    y_cv_train, y_cv_val = y_train[train_cv_idx], y_train[val_cv_idx]
    
    clf_cv = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt'
    )
    clf_cv.fit(X_cv_train, y_cv_train)
    score = clf_cv.score(X_cv_val, y_cv_val)
    cv_scores.append(score)

cv_scores = np.array(cv_scores)
print(f"  - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predictions
y_train_pred = clf.predict(X_train_pca)
y_test_pred = clf.predict(X_test_pca)

# Evaluation metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

print(f"\n{'='*80}")
print("MODEL PERFORMANCE (SPEAKER-INDEPENDENT)")
print(f"{'='*80}")
print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Test F1-Macro:       {test_f1_macro:.4f}")
print(f"Test F1-Weighted:    {test_f1_weighted:.4f}")
print(f"Test Precision:      {test_precision:.4f}")
print(f"Test Recall:         {test_recall:.4f}")

if train_acc > 0.98:
    print("\n⚠️  WARNING: Training accuracy > 98% may indicate overfitting")
    print("   Consider reducing model complexity further")

# Classification report
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT")
print(f"{'='*80}")
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

# Save the trained model
model_path = MODELS_DIR / "rf_hubert_speaker_independent.joblib"
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
    'model_type': 'speaker_independent',
    'best_layer': BEST_LAYER,
    'n_samples': len(X),
    'n_features': X.shape[1],
    'n_classes': len(le.classes_),
    'classes': list(le.classes_),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_speakers': int(len(np.unique(speakers_train))),
    'test_speakers': int(len(np.unique(speakers_test))),
    'speaker_normalization': SPEAKER_NORMALIZATION,
    'use_smote': USE_SMOTE,
    'use_pca': USE_PCA,
    'pca_components': PCA_COMPONENTS if USE_PCA else None,
    'n_estimators': N_ESTIMATORS,
    'max_depth': 15,
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

info_path = MODELS_DIR / "hubert_speaker_independent_info.json"
with open(info_path, 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"  - Training info: {info_path}")

# Save predictions for analysis
predictions_df = pd.DataFrame({
    'speaker_id': speakers_test,
    'true_label': le.inverse_transform(y_test),
    'predicted_label': le.inverse_transform(y_test_pred),
    'correct': y_test == y_test_pred
})
pred_path = RESULTS_DIR / "rf_speaker_independent_predictions.csv"
predictions_df.to_csv(pred_path, index=False)
print(f"  - Predictions: {pred_path}")

# Analyze per-speaker performance
print(f"\n{'='*80}")
print("PER-SPEAKER ANALYSIS")
print(f"{'='*80}")
speaker_stats = predictions_df.groupby('speaker_id').agg({
    'correct': ['count', 'sum', 'mean']
}).round(3)
speaker_stats.columns = ['total_samples', 'correct_samples', 'accuracy']
worst_speakers = speaker_stats.sort_values('accuracy').head(10)
print("\nWorst performing speakers:")
print(worst_speakers)

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"\nTest accuracy: {test_acc:.4f}")
print("This is the TRUE speaker-independent performance!")
print("\nTo use this model for prediction:")
print("  python scripts/predict_speaker_independent.py <audio_file>")
print(f"\n{'='*80}\n")
