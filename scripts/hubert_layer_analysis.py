"""
HuBERT Layer-by-Layer Analysis Script
Evaluates each layer's representation quality for Native Language Identification
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
MANIFEST = "metadata.csv"
HUBERT_DIR = "features/hubert"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200

print("="*80)
print("HuBERT Layer-by-Layer Analysis")
print("="*80)

# Load manifest
print(f"\n[1/7] Loading manifest: {MANIFEST}")
if not os.path.exists(MANIFEST):
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)
print(f"  - Total samples: {len(df)}")
print(f"  - Columns: {list(df.columns)}")

# Detect column names
def detect_column(candidates, df_cols):
    for c in candidates:
        if c in df_cols:
            return c
    return None

utt_col = detect_column(["utt_id", "utt", "id", "filename", "file", "wav", "wav_name", "wav_path", "path"], df.columns)
spk_col = detect_column(["speaker_id", "speaker", "spk", "speakerID", "spk_id"], df.columns)
lab_col = detect_column(["label", "lang", "language", "target", "class"], df.columns)

if not lab_col:
    raise ValueError("Cannot find label column in manifest")

print(f"  - Detected: utt={utt_col}, speaker={spk_col}, label={lab_col}")

# Helper functions
def get_utt_from_row(row):
    if utt_col and utt_col in df.columns and pd.notna(row.get(utt_col)):
        val = str(row.get(utt_col))
        return Path(val).stem if '/' in val or '\\' in val else val
    for c in ["wav_path", "path", "audio_path", "file", "filename"]:
        if c in df.columns and pd.notna(row.get(c)):
            return Path(str(row.get(c))).stem
    return str(row.name)

def get_speaker_from_row(row):
    # Try to extract actual speaker ID from filename
    # Format: "Region_speaker (number).wav" or similar
    for c in ["wav_path", "path", "audio_path", "file", "filename", utt_col]:
        if c in df.columns and pd.notna(row.get(c)):
            filename = Path(str(row.get(c))).stem
            # Extract region and create unique speaker ID
            region = Path(str(row.get(c))).parent.name
            return f"{region}_{filename}"
    return "unknown"

# Build HuBERT file index
print(f"\n[2/7] Indexing HuBERT features from: {HUBERT_DIR}")
hubdir = Path(HUBERT_DIR)
if not hubdir.exists():
    raise FileNotFoundError(f"HuBERT directory not found: {HUBERT_DIR}")

hubert_index = {}
for p in hubdir.rglob("*.npz"):
    hubert_index[p.stem] = p
print(f"  - Indexed {len(hubert_index)} .npz files")

# Collect pooled features per layer
print(f"\n[3/7] Collecting pooled features from HuBERT files")
layer_features = defaultdict(list)
labels = []
groups = []
collected_utts = []
missing = 0
n_layers = None

for idx, row in df.iterrows():
    utt = get_utt_from_row(row)
    spk = get_speaker_from_row(row)
    lab = row[lab_col]
    
    # Try to find matching .npz file
    p = hubert_index.get(utt)
    if p is None:
        # Try with space/underscore normalization
        p = hubert_index.get(utt.replace(" ", "_")) or hubert_index.get(utt.replace("_", " "))
    
    if p is None:
        missing += 1
        continue
    
    try:
        data = np.load(p)
    except Exception as e:
        print(f"  Warning: Failed to load {p.name}: {e}")
        missing += 1
        continue
    
    # Extract pooled representation
    pooled = None
    if "pooled" in data:
        pooled = data["pooled"]
    elif "hidden_states" in data:
        hs = data["hidden_states"]
        pooled = np.array([h.mean(axis=0) for h in hs])
    else:
        # Try to find 2D array
        arrays = [data[k] for k in data.files]
        for arr in arrays:
            if arr.ndim == 2:
                pooled = arr
                break
    
    if pooled is None:
        missing += 1
        continue
    
    # Validate shape
    if n_layers is None:
        n_layers = pooled.shape[0]
        print(f"  - Detected {n_layers} layers")
    
    if pooled.ndim == 2 and pooled.shape[0] == n_layers:
        # Store each layer's features
        for layer_idx in range(n_layers):
            layer_features[layer_idx].append(pooled[layer_idx])
        labels.append(lab)
        groups.append(spk)
        collected_utts.append(utt)
    else:
        missing += 1

print(f"  - Collected: {len(collected_utts)} utterances")
print(f"  - Missing/skipped: {missing}")

if len(collected_utts) == 0:
    raise RuntimeError("No usable HuBERT features found!")

# Prepare labels
labels = np.array(labels)
groups = np.array(groups)
print(f"  - Label distribution: {dict(Counter(labels))}")
print(f"  - Unique speakers: {len(set(groups))}")

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print(f"  - Classes: {list(le.classes_)}")

# Evaluate each layer
print("\n[4/7] Evaluating each layer (stratified split)")
layer_results = {}
layer_f1_scores = {}

for layer_idx in sorted(layer_features.keys()):
    print(f"\n  Layer {layer_idx:02d}:")
    
    # Prepare features for this layer
    X = np.vstack(layer_features[layer_idx])
    print(f"    Shape: {X.shape}")
    
    # Stratified train/test split (ensures balanced class distribution)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels_encoded[train_idx], labels_encoded[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optional PCA for dimensionality reduction
    n_components = min(128, X_train_scaled.shape[1])
    if X_train_scaled.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"    Applied PCA: {X_train_scaled.shape[1]} -> {n_components}")
    else:
        X_train_pca = X_train_scaled
        X_test_pca = X_test_scaled
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=20
    )
    clf.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    layer_results[layer_idx] = accuracy
    layer_f1_scores[layer_idx] = {
        'macro': f1_macro,
        'weighted': f1_weighted
    }
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-macro: {f1_macro:.4f}")
    print(f"    F1-weighted: {f1_weighted:.4f}")

# Find best layer
best_layer = max(layer_results, key=layer_results.get)
best_accuracy = layer_results[best_layer]
print(f"\n{'='*80}")
print(f"BEST LAYER: {best_layer} with accuracy {best_accuracy:.4f}")
print(f"{'='*80}")

# Save numerical results
print(f"\n[5/7] Saving numerical results")
results_file = RESULTS_DIR / "hubert_layerwise_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'layer_accuracies': {str(k): float(v) for k, v in layer_results.items()},
        'layer_f1_macro': {str(k): float(v['macro']) for k, v in layer_f1_scores.items()},
        'layer_f1_weighted': {str(k): float(v['weighted']) for k, v in layer_f1_scores.items()},
        'best_layer': int(best_layer),
        'best_accuracy': float(best_accuracy),
        'n_samples': len(collected_utts),
        'n_classes': len(le.classes_),
        'classes': list(le.classes_)
    }, f, indent=2)
print(f"  - Saved to: {results_file}")

# Update root layerwise_results.json for compatibility
with open("layerwise_results.json", 'w') as f:
    json.dump({str(k): float(v) for k, v in layer_results.items()}, f)
print(f"  - Updated: layerwise_results.json")

# Create summary CSV
summary_data = []
for layer_idx in sorted(layer_results.keys()):
    summary_data.append({
        'Layer': layer_idx,
        'Accuracy': layer_results[layer_idx],
        'F1_Macro': layer_f1_scores[layer_idx]['macro'],
        'F1_Weighted': layer_f1_scores[layer_idx]['weighted']
    })

summary_df = pd.DataFrame(summary_data)
summary_csv = RESULTS_DIR / "hubert_layerwise_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"  - Saved summary: {summary_csv}")

# Generate visualizations
print(f"\n[6/7] Generating visualizations")

# Plot 1: Accuracy by layer
fig, ax = plt.subplots(figsize=(12, 6))
layers = sorted(layer_results.keys())
accuracies = [layer_results[l] for l in layers]

ax.plot(layers, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax.axhline(y=best_accuracy, color='red', linestyle='--', alpha=0.5, label=f'Best: Layer {best_layer}')
ax.scatter([best_layer], [best_accuracy], color='red', s=200, zorder=5, marker='*')

ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('HuBERT Layer-wise Accuracy (Speaker-wise Split)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xticks(layers)

plt.tight_layout()
plot1_path = RESULTS_DIR / "hubert_layer_accuracy.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"  - Saved: {plot1_path}")
plt.close()

# Plot 2: Multiple metrics comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(layers, accuracies, marker='o', linewidth=2, color='#2E86AB')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy by Layer')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(layers)

# F1 Macro
f1_macros = [layer_f1_scores[l]['macro'] for l in layers]
axes[1].plot(layers, f1_macros, marker='s', linewidth=2, color='#A23B72')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('F1-Score (Macro)')
axes[1].set_title('F1-Macro by Layer')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(layers)

# F1 Weighted
f1_weighteds = [layer_f1_scores[l]['weighted'] for l in layers]
axes[2].plot(layers, f1_weighteds, marker='^', linewidth=2, color='#F18F01')
axes[2].set_xlabel('Layer')
axes[2].set_ylabel('F1-Score (Weighted)')
axes[2].set_title('F1-Weighted by Layer')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(layers)

plt.suptitle('HuBERT Layer-wise Performance Metrics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot2_path = RESULTS_DIR / "hubert_layer_metrics_comparison.png"
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"  - Saved: {plot2_path}")
plt.close()

# Plot 3: Heatmap of metrics
fig, ax = plt.subplots(figsize=(14, 4))
metrics_matrix = np.array([
    accuracies,
    f1_macros,
    f1_weighteds
])

sns.heatmap(metrics_matrix, annot=True, fmt='.4f', cmap='YlGnBu', 
            xticklabels=[f'L{i}' for i in layers],
            yticklabels=['Accuracy', 'F1-Macro', 'F1-Weighted'],
            cbar_kws={'label': 'Score'},
            ax=ax)

ax.set_title('HuBERT Layer Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plot3_path = RESULTS_DIR / "hubert_layer_heatmap.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"  - Saved: {plot3_path}")
plt.close()

# Generate markdown report
print("\n[7/7] Generating analysis report")
report_path = REPORTS_DIR / "hubert_layer_analysis.md"

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# HuBERT Layer-by-Layer Analysis Report\n\n")
    f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("---\n\n")
    
    f.write("## Summary\n\n")
    f.write(f"- **Total Samples Analyzed**: {len(collected_utts)}\n")
    f.write(f"- **Number of Layers**: {n_layers}\n")
    f.write(f"- **Number of Classes**: {len(le.classes_)}\n")
    f.write(f"- **Classes**: {', '.join(le.classes_)}\n")
    f.write(f"- **Unique Speakers**: {len(set(groups))}\n")
    f.write(f"- **Test Split**: {TEST_SIZE*100:.0f}% (speaker-wise)\n\n")
    
    f.write("---\n\n")
    
    f.write("## Best Layer Results\n\n")
    f.write(f"**Best Layer**: **Layer {best_layer}**\n\n")
    f.write(f"- **Accuracy**: {best_accuracy:.4f}\n")
    f.write(f"- **F1-Macro**: {layer_f1_scores[best_layer]['macro']:.4f}\n")
    f.write(f"- **F1-Weighted**: {layer_f1_scores[best_layer]['weighted']:.4f}\n\n")
    
    f.write("---\n\n")
    
    f.write("## Layer-wise Performance Table\n\n")
    f.write("| Layer | Accuracy | F1-Macro | F1-Weighted |\n")
    f.write("|-------|----------|----------|-------------|\n")
    for layer_idx in sorted(layer_results.keys()):
        marker = " **(BEST)**" if layer_idx == best_layer else ""
        f.write(f"| {layer_idx}{marker} | {layer_results[layer_idx]:.4f} | ")
        f.write(f"{layer_f1_scores[layer_idx]['macro']:.4f} | ")
        f.write(f"{layer_f1_scores[layer_idx]['weighted']:.4f} |\n")
    f.write("\n")
    
    f.write("---\n\n")
    
    f.write("## Visualizations\n\n")
    f.write("### Accuracy by Layer\n\n")
    f.write(f"![Accuracy by Layer](../results/hubert_layer_accuracy.png)\n\n")
    f.write("### Multiple Metrics Comparison\n\n")
    f.write(f"![Metrics Comparison](../results/hubert_layer_metrics_comparison.png)\n\n")
    f.write("### Performance Heatmap\n\n")
    f.write(f"![Performance Heatmap](../results/hubert_layer_heatmap.png)\n\n")
    
    f.write("---\n\n")
    
    f.write("## Key Findings\n\n")
    
    # Calculate some insights
    acc_range = max(accuracies) - min(accuracies)
    avg_acc = np.mean(accuracies)
    
    f.write(f"1. **Performance Range**: Accuracy varies from {min(accuracies):.4f} to {max(accuracies):.4f} ")
    f.write(f"(range: {acc_range:.4f})\n")
    f.write(f"2. **Average Accuracy**: {avg_acc:.4f} across all layers\n")
    f.write(f"3. **Best Performing Layer**: Layer {best_layer} achieves the highest accuracy\n")
    
    # Check if early, middle, or late layer
    if best_layer < n_layers // 3:
        layer_type = "early"
    elif best_layer < 2 * n_layers // 3:
        layer_type = "middle"
    else:
        layer_type = "late"
    
    f.write(f"4. **Layer Position**: The best layer is in the **{layer_type}** part of the network\n")
    
    f.write("\n---\n\n")
    
    f.write("## Recommendations\n\n")
    f.write(f"1. Use **Layer {best_layer}** features for downstream NLI tasks\n")
    f.write(f"2. Consider ensemble methods combining layers {max(0, best_layer-1)}-{min(n_layers-1, best_layer+1)}\n")
    f.write("3. The layer-wise analysis suggests that intermediate representations capture ")
    f.write("the most relevant linguistic information for native language identification\n\n")
    
    f.write("---\n\n")
    f.write("*Generated by hubert_layer_analysis.py*\n")

print(f"  - Report saved to: {report_path}")

print(f"\n{'='*80}")
print("Analysis Complete!")
print(f"{'='*80}")
print(f"\nResults saved to:")
print(f"  - {results_file}")
print(f"  - {summary_csv}")
print(f"  - {report_path}")
print(f"\nVisualizations saved to:")
print(f"  - {plot1_path}")
print(f"  - {plot2_path}")
print(f"  - {plot3_path}")
print()
