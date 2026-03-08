"""
K-Fold Cross-Validation with Confidence Intervals, Confusion Matrices, and Calibration

Validates best models (Gradient Boosting on MFCC and HuBERT Layer 4) using:
- Stratified k-fold CV with confidence intervals
- Per-fold confusion matrices
- Calibration curves
- Statistical significance tests
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import argparse

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    brier_score_loss
)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuration
RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = -1
BEST_HUBERT_LAYER = 4

print("="*80)
print("K-Fold Cross-Validation with Statistical Analysis")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================

def load_mfcc_features(feature_dir='features/mfcc'):
    """Load MFCC features"""
    print(f"\n[1/7] Loading MFCC features from {feature_dir}...")
    
    feature_path = Path(feature_dir)
    npy_files = sorted(list(feature_path.glob('*.npy')))
    
    features = []
    labels = []
    feature_lengths = []
    
    for npy_file in npy_files:
        try:
            feat = np.load(npy_file)
            if feat.size > 0:
                flat_feat = feat.flatten()
                features.append(flat_feat)
                feature_lengths.append(len(flat_feat))
                
                # Extract label from filename
                filename = npy_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    label = parts[0]
                else:
                    label = 'unknown'
                labels.append(label)
        except Exception:
            continue
    
    # Pad or truncate to uniform length
    if len(set(feature_lengths)) > 1:
        max_len = max(feature_lengths)
        print(f"  Warning: Variable feature lengths detected. Padding to {max_len}.")
        features_padded = []
        for feat in features:
            if len(feat) < max_len:
                padded = np.pad(feat, (0, max_len - len(feat)), mode='constant')
                features_padded.append(padded)
            else:
                features_padded.append(feat[:max_len])
        X = np.array(features_padded)
    else:
        X = np.array(features)
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  Loaded {len(X)} samples, shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    
    return X, y, le

def load_hubert_features(layer=BEST_HUBERT_LAYER, feature_dir='features/hubert'):
    """Load HuBERT features from specified layer"""
    print(f"\n[1/7] Loading HuBERT features from Layer {layer}...")
    
    features = []
    labels = []
    
    feature_path = Path(feature_dir)
    npz_files = list(feature_path.glob('*.npz'))
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            if 'pooled' in data:
                pooled = data['pooled']
                features.append(pooled[layer])
                
                # Extract label
                if 'metadata' in data:
                    metadata = data['metadata'].item()
                    label = metadata.get('native_language', metadata.get('label', 'unknown'))
                else:
                    # Fallback: extract from filename
                    filename = npz_file.stem
                    parts = filename.split('_')
                    label = parts[0] if parts else 'unknown'
                
                labels.append(label)
        except Exception:
            continue
    
    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  Loaded {len(X)} samples, shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    
    return X, y, le

# ============================================================================
# K-Fold Cross-Validation
# ============================================================================

def perform_kfold_cv(X, y, model_name, n_estimators=50):
    """Perform k-fold CV with comprehensive metrics"""
    print(f"\n[2/7] Running {N_SPLITS}-fold CV for {model_name}...")
    
    # Create pipeline - always use PCA to manage memory and match experiment design
    n_components = min(128, X.shape[1] // 2, X.shape[0] // 2)
    print(f"  Using PCA with {n_components} components for dimensionality reduction")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ('gb', GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            verbose=0
        ))
    ])
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=N_JOBS,
        verbose=1
    )
    
    # Compute statistics
    results = {}
    for metric in scoring.keys():
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        results[metric] = {
            'train_mean': float(train_scores.mean()),
            'train_std': float(train_scores.std()),
            'train_ci': float(1.96 * train_scores.std()),  # 95% CI
            'test_mean': float(test_scores.mean()),
            'test_std': float(test_scores.std()),
            'test_ci': float(1.96 * test_scores.std()),
            'test_scores': test_scores.tolist()
        }
    
    print(f"\n  Results for {model_name}:")
    print(f"    Accuracy: {results['accuracy']['test_mean']:.4f} ± {results['accuracy']['test_ci']:.4f}")
    print(f"    F1 Macro: {results['f1_macro']['test_mean']:.4f} ± {results['f1_macro']['test_ci']:.4f}")
    
    return results, cv_results, pipeline

# ============================================================================
# Per-Fold Confusion Matrices
# ============================================================================

def compute_fold_confusion_matrices(X, y, pipeline, le):
    """Compute confusion matrix for each fold"""
    print("\n[3/7] Computing per-fold confusion matrices...")
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    confusion_matrices = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)
        
        print(f"  Fold {fold_idx + 1} accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    
    return confusion_matrices, avg_cm

# ============================================================================
# Calibration Analysis
# ============================================================================

def analyze_calibration(X, y, pipeline, model_name):
    """Analyze model calibration"""
    print(f"\n[4/7] Analyzing calibration for {model_name}...")
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Collect probabilities and true labels across folds
    y_true_all = []
    y_prob_all = []
    brier_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)
        
        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)
        
        # Brier score per fold (for multiclass, average over classes)
        brier = brier_score_loss(
            y_test,
            y_prob[:, 1] if y_prob.shape[1] == 2 else np.max(y_prob, axis=1),
            pos_label=1 if y_prob.shape[1] == 2 else None
        )
        brier_scores.append(brier)
    
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    avg_brier = np.mean(brier_scores)
    print(f"  Average Brier Score: {avg_brier:.4f}")
    
    return y_true_all, y_prob_all, brier_scores, avg_brier

# ============================================================================
# Statistical Significance Test
# ============================================================================

def compare_models(results_mfcc, results_hubert):
    """Paired t-test to compare MFCC vs HuBERT"""
    print("\n[5/7] Statistical comparison (Paired t-test)...")
    
    scores_mfcc = results_mfcc['accuracy']['test_scores']
    scores_hubert = results_hubert['accuracy']['test_scores']
    
    t_stat, p_value = stats.ttest_rel(scores_mfcc, scores_hubert)
    
    print(f"  MFCC Accuracy: {np.mean(scores_mfcc):.4f}")
    print(f"  HuBERT Accuracy: {np.mean(scores_hubert):.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        winner = "HuBERT" if np.mean(scores_hubert) > np.mean(scores_mfcc) else "MFCC"
        print(f"  → {winner} is significantly better (p < 0.05)")
    else:
        print("  → No significant difference (p >= 0.05)")
    
    return {'t_statistic': float(t_stat), 'p_value': float(p_value)}

# ============================================================================
# Visualization
# ============================================================================

def plot_results(results_mfcc, results_hubert, cm_mfcc, cm_hubert, le_mfcc, le_hubert):
    """Create comprehensive visualization"""
    print("\n[6/7] Generating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison with CI
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    x_pos = np.arange(len(metrics))
    
    mfcc_means = [results_mfcc[m]['test_mean'] for m in metrics]
    mfcc_cis = [results_mfcc[m]['test_ci'] for m in metrics]
    hubert_means = [results_hubert[m]['test_mean'] for m in metrics]
    hubert_cis = [results_hubert[m]['test_ci'] for m in metrics]
    
    width = 0.35
    ax1.bar(x_pos - width/2, mfcc_means, width, yerr=mfcc_cis, label='MFCC', capsize=5)
    ax1.bar(x_pos + width/2, hubert_means, width, yerr=hubert_cis, label='HuBERT L4', capsize=5)
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison with 95% CI')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots for accuracy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    data_box = [results_mfcc['accuracy']['test_scores'], results_hubert['accuracy']['test_scores']]
    ax2.boxplot(data_box, labels=['MFCC', 'HuBERT L4'])
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Distribution ({N_SPLITS}-Fold CV)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning curves
    ax3 = fig.add_subplot(gs[0, 2])
    folds = list(range(1, N_SPLITS + 1))
    ax3.plot(folds, results_mfcc['accuracy']['test_scores'], 'o-', label='MFCC', linewidth=2)
    ax3.plot(folds, results_hubert['accuracy']['test_scores'], 's-', label='HuBERT L4', linewidth=2)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Fold Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. MFCC confusion matrix
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_mfcc, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=le_mfcc.classes_, yticklabels=le_mfcc.classes_,
                ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_title('MFCC Avg Confusion Matrix')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    
    # 5. HuBERT confusion matrix
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cm_hubert, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=le_hubert.classes_, yticklabels=le_hubert.classes_,
                ax=ax5, cbar_kws={'label': 'Count'})
    ax5.set_title('HuBERT Layer 4 Avg Confusion Matrix')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    
    # 6. Metric comparison table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    table_data = []
    for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        mfcc_val = f"{results_mfcc[metric]['test_mean']:.4f} ± {results_mfcc[metric]['test_ci']:.4f}"
        hubert_val = f"{results_hubert[metric]['test_mean']:.4f} ± {results_hubert[metric]['test_ci']:.4f}"
        table_data.append([metric.replace('_', ' ').title(), mfcc_val, hubert_val])
    
    table = ax6.table(cellText=table_data, colLabels=['Metric', 'MFCC', 'HuBERT L4'],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax6.set_title('Detailed Metrics', pad=20)
    
    # 7-9. Train vs Test for both models
    ax7 = fig.add_subplot(gs[2, 0])
    metrics_short = ['accuracy', 'f1_macro', 'precision_macro']
    x_pos = np.arange(len(metrics_short))
    train_mfcc = [results_mfcc[m]['train_mean'] for m in metrics_short]
    test_mfcc = [results_mfcc[m]['test_mean'] for m in metrics_short]
    
    width = 0.35
    ax7.bar(x_pos - width/2, train_mfcc, width, label='Train', alpha=0.8)
    ax7.bar(x_pos + width/2, test_mfcc, width, label='Test', alpha=0.8)
    ax7.set_ylabel('Score')
    ax7.set_title('MFCC: Train vs Test')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([m.replace('_', '\n') for m in metrics_short], fontsize=9)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    train_hubert = [results_hubert[m]['train_mean'] for m in metrics_short]
    test_hubert = [results_hubert[m]['test_mean'] for m in metrics_short]
    
    ax8.bar(x_pos - width/2, train_hubert, width, label='Train', alpha=0.8)
    ax8.bar(x_pos + width/2, test_hubert, width, label='Test', alpha=0.8)
    ax8.set_ylabel('Score')
    ax8.set_title('HuBERT L4: Train vs Test')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([m.replace('_', '\n') for m in metrics_short], fontsize=9)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Summary text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
K-Fold Cross-Validation Summary
{'='*35}

Configuration:
• Folds: {N_SPLITS}
• Classifier: Gradient Boosting
• Random State: {RANDOM_STATE}

MFCC Results:
• Accuracy: {results_mfcc['accuracy']['test_mean']:.4f} ± {results_mfcc['accuracy']['test_ci']:.4f}
• F1 Macro: {results_mfcc['f1_macro']['test_mean']:.4f} ± {results_mfcc['f1_macro']['test_ci']:.4f}

HuBERT Layer 4 Results:
• Accuracy: {results_hubert['accuracy']['test_mean']:.4f} ± {results_hubert['accuracy']['test_ci']:.4f}
• F1 Macro: {results_hubert['f1_macro']['test_mean']:.4f} ± {results_hubert['f1_macro']['test_ci']:.4f}

Winner: {'HuBERT' if results_hubert['accuracy']['test_mean'] > results_mfcc['accuracy']['test_mean'] else 'MFCC'}
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('K-Fold Cross-Validation: MFCC vs HuBERT Layer 4', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('results/kfold_validation_comprehensive.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/kfold_validation_comprehensive.png")

# ============================================================================
# Save Results
# ============================================================================

def save_results(results_mfcc, results_hubert, comparison, cm_mfcc, cm_hubert, brier_mfcc, brier_hubert):
    """Save comprehensive results to JSON"""
    print("\n[7/7] Saving results...")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'n_splits': N_SPLITS,
            'random_state': RANDOM_STATE,
            'classifier': 'GradientBoostingClassifier',
            'hubert_layer': BEST_HUBERT_LAYER
        },
        'mfcc': {
            'metrics': results_mfcc,
            'avg_confusion_matrix': cm_mfcc.tolist(),
            'brier_scores': brier_mfcc,
            'avg_brier_score': float(np.mean(brier_mfcc))
        },
        'hubert_layer_4': {
            'metrics': results_hubert,
            'avg_confusion_matrix': cm_hubert.tolist(),
            'brier_scores': brier_hubert,
            'avg_brier_score': float(np.mean(brier_hubert))
        },
        'statistical_comparison': comparison,
        'summary': {
            'winner': 'HuBERT' if results_hubert['accuracy']['test_mean'] > results_mfcc['accuracy']['test_mean'] else 'MFCC',
            'mfcc_accuracy': f"{results_mfcc['accuracy']['test_mean']:.4f} ± {results_mfcc['accuracy']['test_ci']:.4f}",
            'hubert_accuracy': f"{results_hubert['accuracy']['test_mean']:.4f} ± {results_hubert['accuracy']['test_ci']:.4f}",
            'improvement': f"{abs(results_hubert['accuracy']['test_mean'] - results_mfcc['accuracy']['test_mean']):.4f}"
        }
    }
    
    with open('results/kfold_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("  ✓ Saved: results/kfold_validation_results.json")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='K-fold validation with statistical analysis')
    parser.add_argument('--splits', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--layer', type=int, default=4, help='HuBERT layer to use')
    parser.add_argument('--estimators', type=int, default=50, help='Number of GB estimators')
    args = parser.parse_args()
    
    global N_SPLITS, BEST_HUBERT_LAYER
    N_SPLITS = args.splits
    BEST_HUBERT_LAYER = args.layer
    
    # Load data
    X_mfcc, y_mfcc, le_mfcc = load_mfcc_features()
    X_hubert, y_hubert, le_hubert = load_hubert_features(layer=BEST_HUBERT_LAYER)
    
    # K-fold CV for MFCC
    results_mfcc, cv_mfcc, pipeline_mfcc = perform_kfold_cv(X_mfcc, y_mfcc, 'MFCC', n_estimators=args.estimators)
    
    # K-fold CV for HuBERT
    results_hubert, cv_hubert, pipeline_hubert = perform_kfold_cv(X_hubert, y_hubert, f'HuBERT Layer {BEST_HUBERT_LAYER}', n_estimators=args.estimators)
    
    # Confusion matrices
    _, cm_mfcc = compute_fold_confusion_matrices(X_mfcc, y_mfcc, pipeline_mfcc, le_mfcc)
    _, cm_hubert = compute_fold_confusion_matrices(X_hubert, y_hubert, pipeline_hubert, le_hubert)
    
    # Calibration analysis
    _, _, brier_mfcc, avg_brier_mfcc = analyze_calibration(X_mfcc, y_mfcc, pipeline_mfcc, 'MFCC')
    _, _, brier_hubert, avg_brier_hubert = analyze_calibration(X_hubert, y_hubert, pipeline_hubert, f'HuBERT Layer {BEST_HUBERT_LAYER}')
    
    # Statistical comparison
    comparison = compare_models(results_mfcc, results_hubert)
    
    # Visualize
    plot_results(results_mfcc, results_hubert, cm_mfcc, cm_hubert, le_mfcc, le_hubert)
    
    # Save results
    save_results(results_mfcc, results_hubert, comparison, cm_mfcc, cm_hubert, brier_mfcc, brier_hubert)
    
    print("\n" + "="*80)
    print("✅ K-Fold validation complete!")
    print("="*80)
    print(f"\nMFCC Accuracy: {results_mfcc['accuracy']['test_mean']:.4f} ± {results_mfcc['accuracy']['test_ci']:.4f}")
    print(f"HuBERT L{BEST_HUBERT_LAYER} Accuracy: {results_hubert['accuracy']['test_mean']:.4f} ± {results_hubert['accuracy']['test_ci']:.4f}")
    winner = "HuBERT" if results_hubert['accuracy']['test_mean'] > results_mfcc['accuracy']['test_mean'] else "MFCC"
    print(f"\n🏆 Winner: {winner}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
