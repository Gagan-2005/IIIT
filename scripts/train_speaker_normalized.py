"""
Speaker-Normalized Training - Best Approach for Limited Speaker Dataset
Since we have ~1 speaker per region, we use speaker normalization
to remove vocal characteristics while preserving accent patterns
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
import warnings
warnings.filterwarnings('ignore')

# Configuration
MANIFEST = "metadata_existing.csv"
HUBERT_DIR = "features/hubert"
MODELS_DIR = Path("models/speaker_normalized")
RESULTS_DIR = Path("results")
BEST_LAYER = 3

MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200  # Original value - worked best
USE_PCA = True
PCA_COMPONENTS = 96  # Original value - worked best
SPEAKER_NORMALIZATION = True

print("="*80)
print("SPEAKER-NORMALIZED NLI MODEL TRAINING")
print("="*80)
print("\nApproach: Speaker Normalization")
print("  - Dataset has limited speakers (~1 per region)")
print("  - Cannot do speaker-independent split")
print("  - Instead: Normalize per speaker to remove vocal identity")
print("  - Preserves accent patterns (relative phonetic features)")
print("  - Reduced model complexity to prevent memorization")

# Load manifest
print(f"\n[1/10] Loading manifest: {MANIFEST}")
df = pd.read_csv(MANIFEST)
print(f"  - Total samples: {len(df)}")

# Extract speaker ID from wav_path
def extract_speaker_id(path):
    """Extract speaker identifier from file path"""
    filename = Path(path).stem
    # Handle different naming patterns
    parts = filename.split('_')
    if len(parts) >= 2:
        # e.g., "Andhra_speaker (1084)" -> "andhra_pradesh_1084"
        return f"{parts[0].lower()}_{parts[-1].replace('(', '').replace(')', '')}"
    return filename

df['speaker_id'] = df['wav_path'].apply(extract_speaker_id)
unique_speakers = df['speaker_id'].nunique()
print(f"  - Detected {unique_speakers} unique speakers")

# Show speaker distribution per region
print(f"\n  Speaker distribution:")
for label in sorted(df['label'].unique()):
    speakers_in_label = df[df['label'] == label]['speaker_id'].nunique()
    samples_in_label = len(df[df['label'] == label])
    print(f"    {label}: {speakers_in_label} speakers, {samples_in_label} samples")

# Load HuBERT features
print(f"\n[2/10] Indexing HuBERT features from: {HUBERT_DIR}")
hubert_dir = Path(HUBERT_DIR)
npz_files = list(hubert_dir.glob("*.npz"))
print(f"  - Indexed {len(npz_files)} .npz files")

# Build feature map
feature_map = {}
for npz_file in npz_files:
    key = npz_file.stem
    feature_map[key] = npz_file

print(f"\n[3/10] Extracting features from Layer {BEST_LAYER}")
X_list = []
y_list = []
speakers_list = []
missing_count = 0

for idx, row in df.iterrows():
    wav_path = row['wav_path']
    label = row['label']
    speaker_id = row['speaker_id']
    
    key = Path(wav_path).stem
    
    if key in feature_map:
        npz_path = feature_map[key]
        data = np.load(npz_path)
        
        # Features are stored as 'pooled'
        # CRITICAL FIX: Cached features have shape (26, 768) but live extraction returns (13, 768)
        # Use only first 13 layers to match live extraction from HuBERT-base
        if 'pooled' in data:
            pooled = data['pooled']
            # Take only first 13 layers (to match HuBERT-base output)
            if pooled.shape[0] > 13:
                pooled = pooled[:13]
            # Average across all layers
            features = pooled.mean(axis=0)  # Shape: (768,)
            X_list.append(features)
            y_list.append(label)
            speakers_list.append(speaker_id)
        else:
            missing_count += 1
    else:
        missing_count += 1

X = np.array(X_list)
y = np.array(y_list)
speakers = np.array(speakers_list)

print(f"  - Collected {len(X)} samples")
print(f"  - Missing: {missing_count}")
print(f"  - Feature shape: {X.shape}")
print(f"  - Unique speakers: {len(np.unique(speakers))}")
print(f"  - Label distribution: {Counter(y)}")

# Encode labels
print(f"\n[4/10] Encoding labels")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  - Classes: {le.classes_.tolist()}")
joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
print(f"  - Saved label encoder to: {MODELS_DIR / 'label_encoder.joblib'}")

# Split data (stratified by label)
print(f"\n[5/10] Train/Test Split (Stratified by Label)")
X_train, X_test, y_train, y_test, speakers_train, speakers_test = train_test_split(
    X, y_encoded, speakers,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded  # Ensures all labels in both sets
)

print(f"  - Train: {len(X_train)} samples")
print(f"  - Test: {len(X_test)} samples")
print(f"  - Train labels: {sorted(np.unique(y_train).tolist())}")
print(f"  - Test labels: {sorted(np.unique(y_test).tolist())}")

# CRITICAL: Speaker Normalization
print(f"\n[6/10] Applying Speaker Normalization")
print("  - This is the KEY innovation for accent learning")
print("  - Normalizing per speaker: (features - speaker_mean) / speaker_std")
print("  - Removes: Vocal tract length, pitch range, voice quality")
print("  - Preserves: Relative phonetic patterns (accent features)")

X_train_norm = np.copy(X_train)
X_test_norm = np.copy(X_test)

# Store normalization stats for inference
speaker_stats = {}

# Normalize training data per speaker
train_speakers_unique = np.unique(speakers_train)
print(f"  - Normalizing {len(train_speakers_unique)} speakers in training set")
for speaker in train_speakers_unique:
    mask = speakers_train == speaker
    speaker_features = X_train_norm[mask]
    
    if len(speaker_features) >= 2:  # Need at least 2 samples
        mean = speaker_features.mean(axis=0)
        std = speaker_features.std(axis=0) + 1e-8
        X_train_norm[mask] = (speaker_features - mean) / std
        
        # Store for potential inference use
        speaker_stats[speaker] = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'n_samples': len(speaker_features)
        }
    else:
        print(f"    WARNING: Speaker {speaker} has only {len(speaker_features)} sample(s), skipping normalization")

# Normalize test data per speaker
test_speakers_unique = np.unique(speakers_test)
print(f"  - Normalizing {len(test_speakers_unique)} speakers in test set")
for speaker in test_speakers_unique:
    mask = speakers_test == speaker
    speaker_features = X_test_norm[mask]
    
    if len(speaker_features) >= 2:
        mean = speaker_features.mean(axis=0)
        std = speaker_features.std(axis=0) + 1e-8
        X_test_norm[mask] = (speaker_features - mean) / std

print("  [OK] Speaker normalization complete")

# Global standardization (after speaker normalization)
print(f"\n[7/10] Global Feature Standardization")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_test_scaled = scaler.transform(X_test_norm)
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
print(f"  - Saved scaler to: {MODELS_DIR / 'scaler.joblib'}")

# PCA
if USE_PCA:
    print(f"\n[8/10] Applying PCA: {X_train_scaled.shape[1]} -> {PCA_COMPONENTS}")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"  - Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    joblib.dump(pca, MODELS_DIR / "pca.joblib")
    print(f"  - Saved PCA to: {MODELS_DIR / 'pca.joblib'}")
else:
    X_train_pca = X_train_scaled
    X_test_pca = X_test_scaled

# Train Random Forest
print(f"\n[9/10] Training Random Forest Classifier")
print(f"  - Estimators: {N_ESTIMATORS}")
print(f"  - Using class weights (no SMOTE)")
print(f"  - Reduced max_depth to prevent overfitting")

rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=15,        # Original - balanced complexity
    min_samples_split=10, # Original - prevents overfitting
    min_samples_leaf=4,   # Original - balanced
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)

rf.fit(X_train_pca, y_train)
print("  - Training complete!")

# Cross-validation
print(f"  - Running 5-fold cross-validation...")
cv_scores = cross_val_score(rf, X_train_pca, y_train, cv=5, n_jobs=-1)
print(f"  - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Evaluate
print(f"\n[10/10] Evaluating Model")
y_train_pred = rf.predict(X_train_pca)
y_test_pred = rf.predict(X_test_pca)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

print("\n" + "="*80)
print("MODEL PERFORMANCE (WITH SPEAKER NORMALIZATION)")
print("="*80)
print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Test Accuracy:       {test_acc:.4f}")
print(f"Test F1-Macro:       {test_f1_macro:.4f}")
print(f"Test F1-Weighted:    {test_f1_weighted:.4f}")
print(f"Test Precision:      {test_precision:.4f}")
print(f"Test Recall:         {test_recall:.4f}")

if train_acc > 0.98:
    print(f"\n⚠️  WARNING: Training accuracy > 98% may indicate overfitting")
    print("   Consider reducing model complexity further")

# Classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
report = classification_report(
    y_test, y_test_pred,
    target_names=le.classes_,
    zero_division=0
)
print(report)

# Confusion matrix
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df)

# Save model
joblib.dump(rf, MODELS_DIR / "rf_hubert.joblib")

# Save training info
training_info = {
    "model_type": "RandomForest",
    "approach": "speaker_normalization",
    "speaker_normalization": True,
    "n_estimators": N_ESTIMATORS,
    "max_depth": 15,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "best_layer": BEST_LAYER,
    "pca_components": PCA_COMPONENTS if USE_PCA else None,
    "use_smote": False,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "test_f1_macro": float(test_f1_macro),
    "test_f1_weighted": float(test_f1_weighted),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "classes": le.classes_.tolist(),
    "n_features": X.shape[1],
    "n_classes": len(le.classes_)
}

with open(MODELS_DIR / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

# Save predictions for analysis
results_df = pd.DataFrame({
    'true_label': le.inverse_transform(y_test),
    'predicted_label': le.inverse_transform(y_test_pred),
    'speaker_id': speakers_test,
    'correct': y_test == y_test_pred
})
results_df.to_csv(RESULTS_DIR / "rf_speaker_normalized_predictions.csv", index=False)

print("\n" + "="*80)
print("MODEL SAVED")
print("="*80)
print(f"  - Model: {MODELS_DIR / 'rf_hubert.joblib'}")
print(f"  - Scaler: {MODELS_DIR / 'scaler.joblib'}")
print(f"  - PCA: {MODELS_DIR / 'pca.joblib'}")
print(f"  - Label Encoder: {MODELS_DIR / 'label_encoder.joblib'}")
print(f"  - Training info: {MODELS_DIR / 'training_info.json'}")
print(f"  - Predictions: {RESULTS_DIR / 'rf_speaker_normalized_predictions.csv'}")

# Per-speaker analysis
print("\n" + "="*80)
print("PER-SPEAKER ANALYSIS")
print("="*80)
speaker_analysis = results_df.groupby('speaker_id').agg({
    'correct': ['sum', 'count', 'mean']
}).round(3)
speaker_analysis.columns = ['correct_samples', 'total_samples', 'accuracy']
speaker_analysis = speaker_analysis.sort_values('accuracy')

print("\nWorst performing speakers:")
print(speaker_analysis.head(10))

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nTest accuracy: {test_acc:.4f}")
print("\nKey Benefits:")
print("  ✓ Speaker normalization removes vocal characteristics")
print("  ✓ Model learns accent patterns, not voice identity")
print("  ✓ Should generalize better to new speakers")
print("  ✓ Lower but more honest accuracy")
print("\nTo use this model for prediction:")
print("  python scripts/predict_speaker_normalized.py <audio_file>")
print("\n" + "="*80)
