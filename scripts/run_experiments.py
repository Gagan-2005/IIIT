"""
Comprehensive Experiments for NLI Project
==========================================

Experiment 1: HuBERT (best layer) + RF vs MFCC + RF
Experiment 2: MFCC + Multiple Classifiers Comparison
Experiment 3: Best MFCC Classifier + Each HuBERT Layer Comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
import os
from tqdm import tqdm
import time
from pathlib import Path as _Path
import tempfile
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Incremental saving helpers
# ---------------------------------------------------------------------------
_EXP_DIR = _Path('results/experiments')
_EXP_DIR.mkdir(parents=True, exist_ok=True)

def _save_json(rel_path, data):
    """Write JSON atomically to results/experiments/<rel_path>."""
    target = _EXP_DIR / rel_path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=_EXP_DIR)
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def _save_partial(exp1=None, exp2=None, exp3=None):
    """Save a rolling partial summary so interruption doesn't lose progress."""
    payload = {
        'timestamp': datetime.now().isoformat(),
        'experiment_1': exp1,
        'experiment_2': exp2,
        'experiment_3': exp3
    }
    _save_json('summary_partial.json', payload)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_JOBS = -1
BEST_LAYER = 3  # From layer analysis

print("="*80)
print("NLI EXPERIMENTS - Comprehensive Comparison")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_mfcc_features(feature_dir='features/mfcc', cache_file='features/mfcc_cache.npz'):
    """Load MFCC features from .npy files with caching"""
    print("\n[Loading MFCC Features]")
    
    # Check for cached data
    if os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        cache = np.load(cache_file, allow_pickle=True)
        X = cache['features']
        y = cache['labels']
        le = LabelEncoder()
        le.classes_ = cache['classes']
        file_ids = cache['file_ids'].tolist()
        print(f"  ✓ Loaded {len(X)} samples from cache")
        print(f"  Feature shape: {X.shape}")
        print(f"  Classes: {list(le.classes_)}")
        return X, y, le, file_ids
    
    features = []
    labels = []
    file_ids = []
    
    feature_path = Path(feature_dir)
    npy_files = list(feature_path.glob('*.npy'))
    
    print(f"  Found {len(npy_files)} .npy files")
    print("  Loading... (this may take a few minutes)")
    
    # Label mapping from filename patterns
    label_keywords = {
        'andhra': 'andhra_pradesh',
        'gujrat': 'gujrat',
        'jharkhand': 'jharkhand', 
        'karnataka': 'karnataka',
        'kerala': 'kerala',
        'tamil': 'tamil'
    }
    
    skipped = 0
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            
            # Flatten if multi-dimensional
            if len(data.shape) > 1:
                data = data.flatten()
            
            features.append(data)
            
            # Extract label from filename
            filename_lower = npy_file.stem.lower()
            label = None
            for keyword, lang in label_keywords.items():
                if keyword in filename_lower:
                    label = lang
                    break
            
            if label is None:
                skipped += 1
                features.pop()
                continue
                
            labels.append(label)
            file_ids.append(npy_file.stem)
        except Exception:
            skipped += 1
            continue
    
    # Ensure all features have same dimension
    if features:
        min_len = min(len(f) for f in features)
        features = [f[:min_len] for f in features]
    
    X = np.array(features)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  ✓ Loaded {len(X)} samples (skipped {skipped})")
    print(f"  Feature shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    
    # Save cache
    print(f"  Saving cache to {cache_file}...")
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    np.savez_compressed(cache_file, features=X, labels=y, classes=le.classes_, file_ids=np.array(file_ids))
    print("  ✓ Cache saved")
    
    return X, y, le, file_ids

def load_hubert_features(layer=BEST_LAYER, feature_dir='features/hubert'):
    """Load HuBERT features from .npz files"""
    print(f"\n[Loading HuBERT Features - Layer {layer}]")
    
    features = []
    labels = []
    file_ids = []
    
    feature_path = Path(feature_dir)
    npz_files = list(feature_path.glob('*.npz'))
    
    print(f"  Found {len(npz_files)} .npz files")
    
    # Label mapping from filename patterns
    label_keywords = {
        'andhra': 'andhra_pradesh',
        'gujrat': 'gujrat',
        'jharkhand': 'jharkhand', 
        'karnataka': 'karnataka',
        'kerala': 'kerala',
        'tamil': 'tamil'
    }
    
    skipped = 0
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            if 'pooled' in data:
                pooled = data['pooled']
                features.append(pooled[layer])
                
                # Extract label from filename
                filename_lower = npz_file.stem.lower()
                label = None
                for keyword, lang in label_keywords.items():
                    if keyword in filename_lower:
                        label = lang
                        break
                
                if label is None:
                    skipped += 1
                    features.pop()
                    continue
                    
                labels.append(label)
                file_ids.append(npz_file.stem)
        except Exception:
            skipped += 1
            continue
    
    X = np.array(features)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  ✓ Loaded {len(X)} samples (skipped {skipped})")
    print(f"  Feature shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    
    return X, y, le, file_ids

# ============================================================================
# EXPERIMENT 1: HuBERT (Best Layer) + RF vs MFCC + RF
# ============================================================================

def experiment_1():
    """Compare HuBERT best layer vs MFCC with Random Forest"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: HuBERT (Layer 3) + RF vs MFCC + RF")
    print("="*80)
    
    results = {}
    
    # 1. Load MFCC Features
    X_mfcc, y_mfcc, le_mfcc, ids_mfcc = load_mfcc_features()
    X_mfcc_train, X_mfcc_test, y_mfcc_train, y_mfcc_test = train_test_split(
        X_mfcc, y_mfcc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_mfcc
    )
    
    # 2. Load HuBERT Features (Layer 3)
    X_hubert, y_hubert, le_hubert, ids_hubert = load_hubert_features(layer=BEST_LAYER)
    X_hubert_train, X_hubert_test, y_hubert_train, y_hubert_test = train_test_split(
        X_hubert, y_hubert, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_hubert
    )
    
    # 3. Train MFCC + RF
    print("\n[Training MFCC + Random Forest]")
    scaler_mfcc = StandardScaler()
    X_mfcc_train_scaled = scaler_mfcc.fit_transform(X_mfcc_train)
    X_mfcc_test_scaled = scaler_mfcc.transform(X_mfcc_test)
    
    rf_mfcc = RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
        n_jobs=N_JOBS, class_weight='balanced'
    )
    rf_mfcc.fit(X_mfcc_train_scaled, y_mfcc_train)
    
    y_mfcc_pred = rf_mfcc.predict(X_mfcc_test_scaled)
    mfcc_acc = accuracy_score(y_mfcc_test, y_mfcc_pred)
    mfcc_f1_macro = f1_score(y_mfcc_test, y_mfcc_pred, average='macro')
    mfcc_f1_weighted = f1_score(y_mfcc_test, y_mfcc_pred, average='weighted')
    
    print(f"  Accuracy: {mfcc_acc:.4f}")
    print(f"  F1-Macro: {mfcc_f1_macro:.4f}")
    print(f"  F1-Weighted: {mfcc_f1_weighted:.4f}")
    
    results['MFCC + RF'] = {
        'accuracy': float(mfcc_acc),
        'f1_macro': float(mfcc_f1_macro),
        'f1_weighted': float(mfcc_f1_weighted),
        'n_samples': len(X_mfcc),
        'n_train': len(X_mfcc_train),
        'n_test': len(X_mfcc_test)
    }
    
    # 4. Train HuBERT + RF
    print("\n[Training HuBERT (Layer 3) + Random Forest]")
    scaler_hubert = StandardScaler()
    X_hubert_train_scaled = scaler_hubert.fit_transform(X_hubert_train)
    X_hubert_test_scaled = scaler_hubert.transform(X_hubert_test)
    
    # Apply PCA for HuBERT
    pca = PCA(n_components=128, random_state=RANDOM_STATE)
    X_hubert_train_pca = pca.fit_transform(X_hubert_train_scaled)
    X_hubert_test_pca = pca.transform(X_hubert_test_scaled)
    
    rf_hubert = RandomForestClassifier(
        n_estimators=300, max_depth=20, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, class_weight='balanced'
    )
    rf_hubert.fit(X_hubert_train_pca, y_hubert_train)
    
    y_hubert_pred = rf_hubert.predict(X_hubert_test_pca)
    hubert_acc = accuracy_score(y_hubert_test, y_hubert_pred)
    hubert_f1_macro = f1_score(y_hubert_test, y_hubert_pred, average='macro')
    hubert_f1_weighted = f1_score(y_hubert_test, y_hubert_pred, average='weighted')
    
    print(f"  Accuracy: {hubert_acc:.4f}")
    print(f"  F1-Macro: {hubert_f1_macro:.4f}")
    print(f"  F1-Weighted: {hubert_f1_weighted:.4f}")
    
    results['HuBERT (Layer 3) + RF'] = {
        'accuracy': float(hubert_acc),
        'f1_macro': float(hubert_f1_macro),
        'f1_weighted': float(hubert_f1_weighted),
        'n_samples': len(X_hubert),
        'n_train': len(X_hubert_train),
        'n_test': len(X_hubert_test)
    }
    
    # 5. Comparison
    print("\n" + "-"*80)
    print("EXPERIMENT 1 RESULTS:")
    print("-"*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12} {'Samples':<10}")
    print("-"*80)
    for model, metrics in results.items():
        print(f"{model:<30} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} "
              f"{metrics['f1_weighted']:<12.4f} {metrics['n_samples']:<10}")
    
    diff = hubert_acc - mfcc_acc
    winner = "HuBERT" if diff > 0 else "MFCC"
    print("-"*80)
    print(f"Winner: {winner} (Δ Accuracy: {abs(diff):.4f})")
    print("="*80)
    
    return results

# ============================================================================
# EXPERIMENT 2: MFCC + Multiple Classifiers
# ============================================================================

def experiment_2():
    """Compare multiple classifiers on MFCC features"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: MFCC + Multiple Classifiers")
    print("="*80)
    
    # Load MFCC Features
    X, y, le, ids = load_mfcc_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
            n_jobs=N_JOBS, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, verbose=1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced'
        ),
        'SVM (Linear)': SVC(
            kernel='linear', random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, n_jobs=N_JOBS
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n[Training {name}]")
        start_time = time.time()
        
        clf.fit(X_train_scaled, y_train)
        elapsed = time.time() - start_time
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        
        results[name] = {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        }
    
    # Sort by accuracy
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    
    print("\n" + "-"*80)
    print("EXPERIMENT 2 RESULTS (Sorted by Accuracy):")
    print("-"*80)
    print(f"{'Classifier':<25} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12}")
    print("-"*80)
    for name, metrics in sorted_results.items():
        print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} "
              f"{metrics['f1_weighted']:<12.4f}")
    
    best_classifier = list(sorted_results.keys())[0]
    best_acc = sorted_results[best_classifier]['accuracy']
    print("-"*80)
    print(f"Best Classifier: {best_classifier} (Accuracy: {best_acc:.4f})")
    print("="*80)
    
    return sorted_results, best_classifier

# ============================================================================
# EXPERIMENT 3: Best MFCC Classifier + Each HuBERT Layer
# ============================================================================

def experiment_3(best_classifier_name):
    """Compare best MFCC classifier with each HuBERT layer"""
    print("\n" + "="*80)
    print(f"EXPERIMENT 3: Best MFCC Classifier ({best_classifier_name}) + Each HuBERT Layer")
    print("="*80)
    
    # Load MFCC Features
    X_mfcc, y_mfcc, le_mfcc, ids_mfcc = load_mfcc_features()
    X_mfcc_train, X_mfcc_test, y_mfcc_train, y_mfcc_test = train_test_split(
        X_mfcc, y_mfcc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_mfcc
    )
    
    # Scale MFCC
    scaler_mfcc = StandardScaler()
    X_mfcc_train_scaled = scaler_mfcc.fit_transform(X_mfcc_train)
    X_mfcc_test_scaled = scaler_mfcc.transform(X_mfcc_test)
    
    # Train best classifier on MFCC
    print(f"\n[Training {best_classifier_name} on MFCC]")
    
    if best_classifier_name == 'Random Forest':
        clf_mfcc = RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
            n_jobs=N_JOBS, class_weight='balanced'
        )
    elif best_classifier_name == 'Gradient Boosting':
        clf_mfcc = GradientBoostingClassifier(
            n_estimators=50, max_depth=10, random_state=RANDOM_STATE
        )
    elif best_classifier_name == 'Logistic Regression':
        clf_mfcc = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced'
        )
    elif 'SVM' in best_classifier_name:
        kernel = 'linear' if 'Linear' in best_classifier_name else 'rbf'
        clf_mfcc = SVC(kernel=kernel, random_state=RANDOM_STATE, class_weight='balanced')
    else:
        clf_mfcc = RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
            n_jobs=N_JOBS, class_weight='balanced'
        )
    
    clf_mfcc.fit(X_mfcc_train_scaled, y_mfcc_train)
    y_mfcc_pred = clf_mfcc.predict(X_mfcc_test_scaled)
    mfcc_acc = accuracy_score(y_mfcc_test, y_mfcc_pred)
    
    print(f"  MFCC Accuracy: {mfcc_acc:.4f}")
    
    results = {
        'MFCC': {
            'accuracy': float(mfcc_acc),
            'layer': 'N/A',
            'samples': len(X_mfcc)
        }
    }
    
    # Train same classifier on each HuBERT layer
    print(f"\n[Training {best_classifier_name} on Each HuBERT Layer]")
    
    for layer in range(13):
        print(f"\n  Layer {layer}:")
        
        X_hubert, y_hubert, le_hubert, ids_hubert = load_hubert_features(layer=layer)
        X_hubert_train, X_hubert_test, y_hubert_train, y_hubert_test = train_test_split(
            X_hubert, y_hubert, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_hubert
        )
        
        # Scale and PCA
        scaler_hubert = StandardScaler()
        X_hubert_train_scaled = scaler_hubert.fit_transform(X_hubert_train)
        X_hubert_test_scaled = scaler_hubert.transform(X_hubert_test)
        
        pca = PCA(n_components=128, random_state=RANDOM_STATE)
        X_hubert_train_pca = pca.fit_transform(X_hubert_train_scaled)
        X_hubert_test_pca = pca.transform(X_hubert_test_scaled)
        
        # Create classifier
        if best_classifier_name == 'Random Forest':
            clf_hubert = RandomForestClassifier(
                n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
                n_jobs=N_JOBS, class_weight='balanced'
            )
        elif best_classifier_name == 'Gradient Boosting':
            clf_hubert = GradientBoostingClassifier(
                n_estimators=50, max_depth=10, random_state=RANDOM_STATE
            )
        elif best_classifier_name == 'Logistic Regression':
            clf_hubert = LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight='balanced'
            )
        elif 'SVM' in best_classifier_name:
            kernel = 'linear' if 'Linear' in best_classifier_name else 'rbf'
            clf_hubert = SVC(kernel=kernel, random_state=RANDOM_STATE, class_weight='balanced')
        else:
            clf_hubert = RandomForestClassifier(
                n_estimators=300, max_depth=20, random_state=RANDOM_STATE, 
                n_jobs=N_JOBS, class_weight='balanced'
            )
        
        clf_hubert.fit(X_hubert_train_pca, y_hubert_train)
        y_hubert_pred = clf_hubert.predict(X_hubert_test_pca)
        hubert_acc = accuracy_score(y_hubert_test, y_hubert_pred)
        
        print(f"    Accuracy: {hubert_acc:.4f}")
        
        results[f'HuBERT Layer {layer}'] = {
            'accuracy': float(hubert_acc),
            'layer': layer,
            'samples': len(X_hubert)
        }

        # Incremental save after each layer
        _save_json('experiment_3_progress.json', results)
        _save_partial(exp3=results)
    
    # Find best HuBERT layer
    hubert_results = {k: v for k, v in results.items() if k.startswith('HuBERT')}
    best_hubert_layer = max(hubert_results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n" + "-"*80)
    print("EXPERIMENT 3 RESULTS:")
    print("-"*80)
    print(f"{'Configuration':<25} {'Accuracy':<12} {'Samples':<10}")
    print("-"*80)
    print(f"{'MFCC':<25} {results['MFCC']['accuracy']:<12.4f} {results['MFCC']['samples']:<10}")
    print("-"*80)
    for layer in range(13):
        key = f'HuBERT Layer {layer}'
        acc = results[key]['accuracy']
        samples = results[key]['samples']
        marker = " ★" if key == best_hubert_layer[0] else ""
        print(f"{key:<25} {acc:<12.4f} {samples:<10}{marker}")
    print("-"*80)
    print(f"Best HuBERT Layer: {best_hubert_layer[1]['layer']} "
          f"(Accuracy: {best_hubert_layer[1]['accuracy']:.4f})")
    print(f"MFCC Baseline: {results['MFCC']['accuracy']:.4f}")
    print("="*80)

    # Final experiment 3 save
    _save_json('experiment_3_final.json', results)
    _save_partial(exp3=results)
    
    return results

# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(exp1_results, exp2_results, exp3_results):
    """Create comparison plots"""
    print("\n[Creating Visualizations]")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Experiment 1: HuBERT vs MFCC
    ax1 = plt.subplot(2, 2, 1)
    models = list(exp1_results.keys())
    accuracies = [exp1_results[m]['accuracy'] for m in models]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Experiment 1: HuBERT vs MFCC (Random Forest)', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Experiment 2: Multiple Classifiers
    ax2 = plt.subplot(2, 2, 2)
    clf_names = list(exp2_results.keys())
    clf_accs = [exp2_results[c]['accuracy'] for c in clf_names]
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(clf_names)))
    bars2 = ax2.barh(clf_names, clf_accs, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Experiment 2: MFCC + Multiple Classifiers', fontsize=14, fontweight='bold')
    ax2.set_xlim([0.8, 1.0])
    ax2.grid(axis='x', alpha=0.3)
    for bar, acc in zip(bars2, clf_accs):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{acc:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Experiment 3: Layer-wise comparison
    ax3 = plt.subplot(2, 2, 3)
    layers = ['MFCC'] + [f'L{i}' for i in range(13)]
    layer_accs = [exp3_results['MFCC']['accuracy']] + \
                 [exp3_results[f'HuBERT Layer {i}']['accuracy'] for i in range(13)]
    colors3 = ['red'] + ['blue']*13
    ax3.plot(range(len(layers)), layer_accs, 'o-', linewidth=2, markersize=8)
    ax3.axhline(y=exp3_results['MFCC']['accuracy'], color='red', linestyle='--', 
                linewidth=2, alpha=0.5, label='MFCC Baseline')
    ax3.set_xticks(range(len(layers)))
    ax3.set_xticklabels(layers, rotation=45)
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax3.set_title('Experiment 3: MFCC vs HuBERT Layers', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_data = [
        ['Experiment', 'Best Configuration', 'Accuracy'],
        ['1. Feature Comparison', 
         'HuBERT' if exp1_results['HuBERT (Layer 3) + RF']['accuracy'] > exp1_results['MFCC + RF']['accuracy'] else 'MFCC',
         f"{max(exp1_results[k]['accuracy'] for k in exp1_results):.4f}"],
        ['2. Classifier Comparison', 
         list(exp2_results.keys())[0],
         f"{list(exp2_results.values())[0]['accuracy']:.4f}"],
        ['3. Layer-wise Analysis',
         f"Layer {max(((k, v) for k, v in exp3_results.items() if 'Layer' in k), key=lambda x: x[1]['accuracy'])[1]['layer']}",
         f"{max(v['accuracy'] for k, v in exp3_results.items() if 'Layer' in k):.4f}"]
    ]
    
    table = ax4.table(cellText=summary_data, loc='center', cellLoc='left',
                     colWidths=[0.3, 0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Experimental Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_experiments.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/comprehensive_experiments.png")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all experiments"""
    
    # Experiment 1
    exp1_results = experiment_1()
    _save_json('experiment_1.json', exp1_results)
    _save_partial(exp1=exp1_results)
    
    # Experiment 2
    exp2_results, best_classifier = experiment_2()
    _save_json('experiment_2.json', exp2_results)
    _save_partial(exp1=exp1_results, exp2=exp2_results)
    
    # Experiment 3
    exp3_results = experiment_3(best_classifier)
    _save_partial(exp1=exp1_results, exp2=exp2_results, exp3=exp3_results)
    
    # Create visualizations
    create_visualizations(exp1_results, exp2_results, exp3_results)
    
    # Save all results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_1': exp1_results,
        'experiment_2': exp2_results,
        'experiment_3': exp3_results
    }
    
    with open('results/comprehensive_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Saved: results/comprehensive_experiments.json")
    _save_json('experiments_summary_final.json', all_results)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
