"""
Systematic Hyperparameter Tuning with Grid Search and Cross-Validation
Implements comprehensive tuning for Random Forest baseline

Enhancements:
- Auto-select HuBERT best layer from results/comprehensive_experiments.json (--auto-layer)
- Manual override for layer via --layer
- Fast mode (--fast): fewer params, fewer CV folds, randomized search only by default
- Random-only mode (--random-only) to skip grid search explicitly
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint
import json
from datetime import datetime

# Configuration (defaults; can be overridden by CLI)
BEST_LAYER = 3
N_SPLITS = 5  # K-fold cross-validation
RANDOM_STATE = 42
N_JOBS = -1  # Use all CPUs
FAST_MODE = False
RANDOM_ONLY = False
RANDOM_N_ITER = 100

print("="*80)
print("Systematic Hyperparameter Tuning with Cross-Validation")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================

def load_hubert_features(layer=BEST_LAYER, feature_dir='features/hubert'):
    """Load HuBERT features from .npz files"""
    print(f"\n[1/6] Loading HuBERT features from Layer {layer}...")
    
    features = []
    labels = []
    file_ids = []
    
    feature_path = Path(feature_dir)
    npz_files = list(feature_path.glob('*.npz'))
    
    print(f"  Found {len(npz_files)} .npz files")
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract pooled features for specified layer
            if 'pooled' in data:
                pooled = data['pooled']
                features.append(pooled[layer])
                
                # Extract label from metadata
                metadata = data['metadata'].item()
                label = metadata.get('native_language', metadata.get('label', 'unknown'))
                labels.append(label)
                file_ids.append(npz_file.stem)
        except Exception as e:
            print(f"  Warning: Could not load {npz_file.name}: {e}")
            continue
    
    X = np.array(features)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  Loaded {len(X)} samples")
    print(f"  Feature shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    print("  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(le.inverse_transform(unique), counts):
        print(f"    {cls}: {count}")
    
    return X, y, le, file_ids

# ============================================================================
# Grid Search Configuration
# ============================================================================

# Define search spaces (defaults; may be tightened in FAST mode)
GRID_SEARCH_PARAMS = {
    'rf__n_estimators': [100, 200, 300, 400],
    'rf__max_depth': [10, 20, 30, None],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', None],
    'rf__class_weight': ['balanced', 'balanced_subsample'],
    'pca__n_components': [64, 128, 256]
}

RANDOM_SEARCH_PARAMS = {
    'rf__n_estimators': randint(100, 500),
    'rf__max_depth': [10, 20, 30, 40, None],
    'rf__min_samples_split': randint(2, 20),
    'rf__min_samples_leaf': randint(1, 10),
    'rf__max_features': ['sqrt', 'log2', 0.5, None],
    'rf__class_weight': ['balanced', 'balanced_subsample'],
    'rf__bootstrap': [True, False],
    'pca__n_components': randint(64, 300)
}

def apply_fast_mode():
    """Tighten search spaces and CV for faster runs."""
    global GRID_SEARCH_PARAMS, RANDOM_SEARCH_PARAMS, N_SPLITS, RANDOM_N_ITER, RANDOM_ONLY
    N_SPLITS = 3
    RANDOM_N_ITER = 30
    # Smaller grids
    GRID_SEARCH_PARAMS = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__class_weight': ['balanced'],
        'pca__n_components': [128]
    }
    # Fewer random iters via RANDOM_N_ITER; keep broad priors
    RANDOM_ONLY = True  # prioritize randomized search in fast mode

# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def create_pipeline():
    """Create pipeline with preprocessing and classifier"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
    ])

def run_grid_search(X, y):
    """Run exhaustive grid search"""
    print("\n[2/6] Running Grid Search (this may take a while)...")
    print(f"  Parameter combinations: {np.prod([len(v) for v in GRID_SEARCH_PARAMS.values()])}")
    
    pipeline = create_pipeline()
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        pipeline,
        GRID_SEARCH_PARAMS,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search

def run_random_search(X, y, n_iter=100):
    """Run randomized search"""
    print(f"\n[3/6] Running Randomized Search ({n_iter} iterations)...")
    
    pipeline = create_pipeline()
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        pipeline,
        RANDOM_SEARCH_PARAMS,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    random_search.fit(X, y)
    
    print(f"\n  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.4f}")
    
    return random_search

def cross_validate_model(X, y, params):
    """Perform cross-validation with specific parameters"""
    print("\n[4/6] Running cross-validation with best parameters...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=params.get('pca__n_components', 128), 
                    random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(
            n_estimators=params.get('rf__n_estimators', 200),
            max_depth=params.get('rf__max_depth', None),
            min_samples_split=params.get('rf__min_samples_split', 2),
            min_samples_leaf=params.get('rf__min_samples_leaf', 1),
            max_features=params.get('rf__max_features', 'sqrt'),
            class_weight=params.get('rf__class_weight', 'balanced'),
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Multiple scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }
    
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=N_JOBS
    )
    
    print("\n  Cross-validation results:")
    for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        print(f"    {metric}:")
        print(f"      Train: {train_scores.mean():.4f} (+/- {train_scores.std()*2:.4f})")
        print(f"      Test:  {test_scores.mean():.4f} (+/- {test_scores.std()*2:.4f})")
    
    return cv_results, pipeline

# ============================================================================
# Visualization
# ============================================================================

def plot_search_results(search_results, title, filename):
    """Plot grid search results"""
    print(f"\n[5/6] Generating visualization: {filename}...")
    
    results_df = pd.DataFrame(search_results.cv_results_)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top parameter importance
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    if len(param_cols) > 0:
        # Calculate variance in scores for each parameter
        param_importance = {}
        for param in param_cols:
            # Group by parameter and calculate score variance
            grouped = results_df.groupby(param)['mean_test_score'].agg(['mean', 'std'])
            param_importance[param.replace('param_', '')] = grouped['std'].mean()
        
        # Plot top parameters
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        params, importance = zip(*sorted_params)
        axes[0, 0].barh(params, importance)
        axes[0, 0].set_xlabel('Score Variance')
        axes[0, 0].set_title('Most Influential Parameters')
    
    # 2. Train vs Test scores
    axes[0, 1].scatter(results_df['mean_train_score'], 
                       results_df['mean_test_score'],
                       alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0, 1].set_xlabel('Train Score')
    axes[0, 1].set_ylabel('Test Score')
    axes[0, 1].set_title('Train vs Test Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Score distribution
    axes[1, 0].hist(results_df['mean_test_score'], bins=30, edgecolor='black')
    axes[1, 0].axvline(search_results.best_score_, color='r', 
                       linestyle='--', linewidth=2, label='Best Score')
    axes[1, 0].set_xlabel('CV Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Cross-Validation Score Distribution')
    axes[1, 0].legend()
    
    # 4. Top 10 configurations
    top_10 = results_df.nlargest(10, 'mean_test_score')[['mean_test_score', 'std_test_score']]
    axes[1, 1].barh(range(10), top_10['mean_test_score'], 
                    xerr=top_10['std_test_score'])
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels([f'Config {i+1}' for i in range(10)])
    axes[1, 1].set_xlabel('CV Score')
    axes[1, 1].set_title('Top 10 Configurations')
    axes[1, 1].invert_yaxis()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: results/{filename}")

def plot_learning_curves(cv_results, filename='learning_curves.png'):
    """Plot learning curves from cross-validation"""
    print(f"\n[6/6] Generating learning curves: {filename}...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    titles = ['Accuracy', 'F1 Macro', 'F1 Weighted']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        train_scores = cv_results[f'train_{metric}']
        test_scores = cv_results[f'test_{metric}']
        
        axes[idx].plot(range(1, len(train_scores)+1), train_scores, 
                       'o-', label='Train', linewidth=2)
        axes[idx].plot(range(1, len(test_scores)+1), test_scores, 
                       's-', label='Validation', linewidth=2)
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel(title)
        axes[idx].set_title(f'{title} Across Folds')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: results/{filename}")

# ============================================================================
# Save Results
# ============================================================================

def save_tuning_results(grid_results, random_results, cv_results, best_params):
    """Save comprehensive tuning results"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_layer': BEST_LAYER,
        'cv_folds': N_SPLITS,
        'grid_search': (
            None if grid_results is None else {
                'best_score': float(grid_results.best_score_),
                'best_params': {k: str(v) for k, v in grid_results.best_params_.items()},
                'n_combinations': len(grid_results.cv_results_['params'])
            }
        ),
        'random_search': (
            None if random_results is None else {
                'best_score': float(random_results.best_score_),
                'best_params': {k: str(v) for k, v in random_results.best_params_.items()},
                'n_iterations': len(random_results.cv_results_['params'])
            }
        ),
        'cross_validation': {
            'accuracy_mean': float(cv_results['test_accuracy'].mean()),
            'accuracy_std': float(cv_results['test_accuracy'].std()),
            'f1_macro_mean': float(cv_results['test_f1_macro'].mean()),
            'f1_macro_std': float(cv_results['test_f1_macro'].std()),
            'f1_weighted_mean': float(cv_results['test_f1_weighted'].mean()),
            'f1_weighted_std': float(cv_results['test_f1_weighted'].std())
        },
        'final_params': {k: str(v) for k, v in best_params.items()}
    }
    
    with open('results/hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Saved: results/hyperparameter_tuning_results.json")
    
    # Save detailed results as CSV
    if grid_results is not None:
        grid_df = pd.DataFrame(grid_results.cv_results_)
        grid_df.to_csv('results/grid_search_detailed.csv', index=False)
        print("✓ Saved: results/grid_search_detailed.csv")
    
    if random_results is not None:
        random_df = pd.DataFrame(random_results.cv_results_)
        random_df.to_csv('results/random_search_detailed.csv', index=False)
        print("✓ Saved: results/random_search_detailed.csv")

# ============================================================================
# Utilities: Auto-detect best layer
# ============================================================================

def read_best_layer_from_results(results_path: Path) -> int | None:
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        layer = (
            data.get('winners', {})
                .get('experiment_3', {})
                .get('layer', None)
        )
        if isinstance(layer, int):
            return layer
    except Exception as e:
        print(f"  Warning: Could not read best layer from {results_path}: {e}")
    return None

# ============================================================================
# Main Execution
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='RF hyperparameter tuning on HuBERT features')
    parser.add_argument('--layer', type=int, help='HuBERT layer index to use')
    parser.add_argument('--auto-layer', action='store_true', help='Read best layer from results/comprehensive_experiments.json')
    parser.add_argument('--fast', action='store_true', help='Fast mode: reduced search + 3-fold CV + randomized search only')
    parser.add_argument('--random-only', action='store_true', help='Skip grid search and only run randomized search')
    parser.add_argument('--splits', type=int, help='Override number of CV folds')
    parser.add_argument('--random-iters', type=int, help='Override randomized search iterations')
    return parser.parse_args()

def main():
    global BEST_LAYER, N_SPLITS, FAST_MODE, RANDOM_ONLY, RANDOM_N_ITER

    args = parse_args()

    # Fast mode adjustments
    if args.fast:
        FAST_MODE = True
        apply_fast_mode()
        print("\n[FAST MODE] Enabled: 3-fold CV, randomized search prioritized")

    # Explicit overrides
    if args.splits:
        N_SPLITS = int(args.splits)
    if args.random_only:
        RANDOM_ONLY = True
    if args.random_iters:
        RANDOM_N_ITER = int(args.random_iters)

    # Layer selection
    best_layer_auto = None
    if args.auto_layer:
        best_layer_auto = read_best_layer_from_results(Path('results/comprehensive_experiments.json'))
    if args.layer is not None:
        BEST_LAYER = int(args.layer)
    elif best_layer_auto is not None:
        BEST_LAYER = best_layer_auto

    print(f"\nUsing HuBERT Layer: {BEST_LAYER}")
    print(f"CV folds: {N_SPLITS}; Random iters: {RANDOM_N_ITER}; Random-only: {RANDOM_ONLY}")

    # Load data
    X, y, le, file_ids = load_hubert_features(layer=BEST_LAYER)

    grid_results = None
    random_results = None

    # Optionally run grid search
    if not RANDOM_ONLY:
        grid_results = run_grid_search(X, y)
        plot_search_results(grid_results, 'Grid Search Results', 'grid_search_results.png')

    # Run randomized search
    random_results = run_random_search(X, y, n_iter=RANDOM_N_ITER)
    plot_search_results(random_results, 'Random Search Results', 'random_search_results.png')

    # Choose best parameters
    if grid_results is None or (random_results is not None and random_results.best_score_ >= grid_results.best_score_):
        best_params = random_results.best_params_
        print("\n  Randomized Search selected as best.")
    else:
        best_params = grid_results.best_params_
        print("\n  Grid Search selected as best.")

    # Cross-validate with best parameters
    cv_results, best_pipeline = cross_validate_model(X, y, best_params)
    plot_learning_curves(cv_results)

    # Train final model with best parameters
    print("\n[Final] Training final model with best parameters...")
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, 'models/rf_hubert_optimized.joblib')
    print("  ✓ Saved: models/rf_hubert_optimized.joblib")

    # Save all results
    save_tuning_results(grid_results, random_results, cv_results, best_params)

    print("\n" + "="*80)
    print("✅ Hyperparameter tuning complete!")
    print("="*80)
    final_best_score = max(
        [s for s in [
            grid_results.best_score_ if grid_results is not None else None,
            random_results.best_score_ if random_results is not None else None
        ] if s is not None]
    )
    print(f"\nBest CV Accuracy: {final_best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
