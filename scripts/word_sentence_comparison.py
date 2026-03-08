"""
Word-Level vs Sentence-Level Accent Detection Comparison

Segments audio into words, extracts features, and compares:
- Word-level prediction accuracy and stability
- Sentence-level (full audio) prediction accuracy
- Aggregated word predictions vs direct sentence predictions
- Robustness analysis for MFCC and HuBERT representations
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
from datetime import datetime
import argparse
from collections import Counter
from tqdm import tqdm

import joblib
import torch
from transformers import Wav2Vec2Processor, HubertModel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration
SAMPLE_RATE = 16000
MIN_WORD_DURATION = 0.3  # seconds
MAX_WORD_DURATION = 3.0  # seconds
SILENCE_THRESHOLD = 0.02  # amplitude threshold for silence
MIN_SILENCE_DURATION = 0.2  # seconds
BEST_HUBERT_LAYER = 4

print("="*80)
print("Word-Level vs Sentence-Level Accent Detection")
print("="*80)

# ============================================================================
# Audio Segmentation
# ============================================================================

def segment_audio_to_words(audio_path, sr=SAMPLE_RATE):
    """
    Segment audio into word-level chunks using energy-based VAD
    
    Returns:
        List of (start_time, end_time, audio_segment) tuples
    """
    try:
        audio, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"  Error loading {audio_path}: {e}")
        return []
    
    # Compute energy envelope
    hop_length = 512
    frame_length = 2048
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize energy
    energy = energy / (np.max(energy) + 1e-8)
    
    # Find voice activity (energy above threshold)
    voice_frames = energy > SILENCE_THRESHOLD
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(voice_frames)), sr=sr, hop_length=hop_length)
    
    # Find segments (groups of consecutive voice frames)
    segments = []
    in_segment = False
    start_time = 0
    
    for i, is_voice in enumerate(voice_frames):
        if is_voice and not in_segment:
            # Start of new segment
            start_time = times[i]
            in_segment = True
        elif not is_voice and in_segment:
            # End of segment
            end_time = times[i]
            duration = end_time - start_time
            
            # Filter by duration
            if MIN_WORD_DURATION <= duration <= MAX_WORD_DURATION:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]
                segments.append((start_time, end_time, segment_audio))
            
            in_segment = False
    
    # Handle last segment if audio ends while speaking
    if in_segment and len(times) > 0:
        end_time = times[-1]
        duration = end_time - start_time
        if MIN_WORD_DURATION <= duration <= MAX_WORD_DURATION:
            start_sample = int(start_time * sr)
            segment_audio = audio[start_sample:]
            segments.append((start_time, end_time, segment_audio))
    
    return segments

# ============================================================================
# Feature Extraction
# ============================================================================

def extract_mfcc_from_audio(audio, sr=SAMPLE_RATE, n_mfcc=40):
    """Extract MFCC+delta+delta2 means to match 120-dim training features."""
    try:
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        # Aggregate over time (means only: 40*3 = 120)
        feat = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])
        return feat.astype(np.float32)
    except Exception:
        return None

def extract_hubert_from_audio(audio, processor, model, layer=BEST_HUBERT_LAYER, device='cpu'):
    """Extract HuBERT embeddings from audio segment"""
    try:
        # Ensure audio is correct length
        if len(audio) < 400:  # Minimum length for HuBERT
            audio = np.pad(audio, (0, 400 - len(audio)))
        
        # Process audio
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Get specified layer and pool
            layer_output = hidden_states[layer]
            pooled = torch.mean(layer_output, dim=1).cpu().numpy()[0]
            
        return pooled
    except Exception:
        return None

# ============================================================================
# Prediction Pipeline
# ============================================================================

def load_models():
    """Load trained models and preprocessing objects"""
    print("\n[1/6] Loading trained models...")
    
    models = {}
    
    # Load MFCC model (Random Forest baseline trained on MFCC)
    try:
        models['mfcc_model'] = joblib.load('models/rf_mfcc.joblib')
        models['mfcc_scaler'] = joblib.load('models/scaler.joblib')
        models['mfcc_label_encoder'] = joblib.load('models/label_encoder.joblib')
        print("  ✓ Loaded MFCC model (RF, 120-d input)")
    except Exception as e:
        print(f"  Warning: Could not load MFCC model: {e}")
        models['mfcc_model'] = None
    
    # Load HuBERT model
    try:
        models['hubert_model'] = joblib.load('models/hubert_bestlayer_rf_layer8.joblib')
        models['hubert_scaler'] = joblib.load('models/scaler_hubert.joblib')
        models['hubert_pca'] = joblib.load('models/pca_hubert.joblib')
        models['hubert_label_encoder'] = joblib.load('models/label_encoder.joblib')
        print("  ✓ Loaded HuBERT model (RF, Layer 8)")
    except Exception as e:
        print(f"  Warning: Could not load HuBERT model: {e}")
        models['hubert_model'] = None
    
    # Load HuBERT processor and model for feature extraction
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models['hubert_processor'] = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        models['hubert_feature_model'] = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        models['device'] = device
        print(f"  ✓ Loaded HuBERT feature extractor (device: {device})")
    except Exception as e:
        print(f"  Warning: Could not load HuBERT feature extractor: {e}")
        models['hubert_processor'] = None
        models['hubert_feature_model'] = None
        models['device'] = 'cpu'
    
    return models

def predict_word_level(audio_segments, models, feature_type='mfcc'):
    """
    Predict accent for each word-level segment
    
    Returns:
        List of (prediction, confidence, features) tuples
    """
    predictions = []
    
    for start, end, audio in audio_segments:
        if feature_type == 'mfcc':
            # Extract MFCC
            features = extract_mfcc_from_audio(audio)
            if features is None:
                continue
            
            # Predict
            if models['mfcc_model'] is not None:
                features_scaled = models['mfcc_scaler'].transform([features])
                pred = models['mfcc_model'].predict(features_scaled)[0]
                proba = models['mfcc_model'].predict_proba(features_scaled)[0]
                confidence = np.max(proba)
                label = models['mfcc_label_encoder'].inverse_transform([pred])[0]
                predictions.append((label, confidence, features))
        
        elif feature_type == 'hubert':
            # Extract HuBERT
            if models.get('hubert_processor') is None or models.get('hubert_feature_model') is None:
                continue
            features = extract_hubert_from_audio(
                audio,
                models['hubert_processor'],
                models['hubert_feature_model'],
                layer=BEST_HUBERT_LAYER,
                device=models['device']
            )
            if features is None:
                continue
            
            # Predict
            if models['hubert_model'] is not None:
                features_scaled = models['hubert_scaler'].transform([features])
                features_pca = models['hubert_pca'].transform(features_scaled)
                pred = models['hubert_model'].predict(features_pca)[0]
                proba = models['hubert_model'].predict_proba(features_pca)[0]
                confidence = np.max(proba)
                label = models['hubert_label_encoder'].inverse_transform([pred])[0]
                predictions.append((label, confidence, features))
    
    return predictions

def aggregate_word_predictions(word_predictions):
    """
    Aggregate word-level predictions to sentence-level
    
    Methods:
    - Majority voting
    - Weighted by confidence
    - Average confidence
    """
    if not word_predictions:
        return None, 0, {}
    
    labels = [p[0] for p in word_predictions]
    confidences = [p[1] for p in word_predictions]
    
    # Majority vote
    label_counts = Counter(labels)
    majority_label = label_counts.most_common(1)[0][0]
    
    # Weighted vote
    label_weights = {}
    for label, conf in zip(labels, confidences):
        label_weights[label] = label_weights.get(label, 0) + conf
    weighted_label = max(label_weights, key=label_weights.get)
    
    # Average confidence
    avg_confidence = np.mean(confidences)
    
    aggregation_info = {
        'majority_vote': majority_label,
        'weighted_vote': weighted_label,
        'avg_confidence': avg_confidence,
        'n_words': len(word_predictions),
        'label_distribution': dict(label_counts),
        'confidence_std': np.std(confidences)
    }
    
    return weighted_label, avg_confidence, aggregation_info

def predict_sentence_level(audio_path, models, feature_type='mfcc'):
    """Predict accent from full sentence audio"""
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception:
        return None, 0
    
    if feature_type == 'mfcc':
        features = extract_mfcc_from_audio(audio)
        if features is None:
            return None, 0
        
        if models['mfcc_model'] is not None:
            features_scaled = models['mfcc_scaler'].transform([features])
            pred = models['mfcc_model'].predict(features_scaled)[0]
            proba = models['mfcc_model'].predict_proba(features_scaled)[0]
            confidence = np.max(proba)
            label = models['mfcc_label_encoder'].inverse_transform([pred])[0]
            return label, confidence
    
    elif feature_type == 'hubert':
        features = extract_hubert_from_audio(
            audio,
            models['hubert_processor'],
            models['hubert_feature_model'],
            layer=BEST_HUBERT_LAYER,
            device=models['device']
        )
        if features is None:
            return None, 0
        
        if models['hubert_model'] is not None:
            features_scaled = models['hubert_scaler'].transform([features])
            features_pca = models['hubert_pca'].transform(features_scaled)
            pred = models['hubert_model'].predict(features_pca)[0]
            proba = models['hubert_model'].predict_proba(features_pca)[0]
            confidence = np.max(proba)
            label = models['hubert_label_encoder'].inverse_transform([pred])[0]
            return label, confidence
    
    return None, 0

# ============================================================================
# Comparison and Analysis
# ============================================================================

def process_dataset(data_dir, models, feature_type='mfcc', max_files=None):
    """
    Process all audio files and compare word vs sentence level predictions
    """
    print(f"\n[2/6] Processing dataset with {feature_type.upper()} features...")
    
    data_path = Path(data_dir)
    audio_files = []
    
    # Collect audio files
    for region_dir in data_path.iterdir():
        if region_dir.is_dir():
            audio_files.extend(list(region_dir.glob('*.wav')))
    
    if max_files:
        audio_files = audio_files[:max_files]
    
    print(f"  Found {len(audio_files)} audio files")
    
    results = []
    
    for audio_file in tqdm(audio_files, desc=f"  {feature_type.upper()}"):
        # Extract true label from path
        true_label = audio_file.parent.name
        
        # Segment into words
        word_segments = segment_audio_to_words(audio_file)
        
        if len(word_segments) < 2:  # Skip if too few words
            continue
        
        # Word-level predictions
        word_preds = predict_word_level(word_segments, models, feature_type)
        
        if not word_preds:
            continue
        
        # Aggregate word predictions
        agg_label, agg_conf, agg_info = aggregate_word_predictions(word_preds)
        
        # Sentence-level prediction
        sent_label, sent_conf = predict_sentence_level(audio_file, models, feature_type)
        
        if sent_label is None:
            continue
        
        results.append({
            'file': audio_file.name,
            'true_label': true_label,
            'word_aggregated_label': agg_label,
            'word_aggregated_confidence': agg_conf,
            'word_confidence_std': agg_info['confidence_std'],
            'n_words': agg_info['n_words'],
            'sentence_label': sent_label,
            'sentence_confidence': sent_conf,
            'word_correct': (agg_label == true_label),
            'sentence_correct': (sent_label == true_label),
            'agreement': (agg_label == sent_label)
        })
    
    return results

def analyze_results(results_mfcc, results_hubert):
    """Comprehensive analysis of word vs sentence level predictions"""
    print("\n[3/6] Analyzing results...")
    
    analysis = {}
    
    for feature_type, results in [('MFCC', results_mfcc), ('HuBERT', results_hubert)]:
        if not results:
            continue
        
        df = pd.DataFrame(results)
        
        # Accuracy metrics
        word_acc = df['word_correct'].mean()
        sent_acc = df['sentence_correct'].mean()
        agreement_rate = df['agreement'].mean()
        
        # Confidence analysis
        word_conf_mean = df['word_aggregated_confidence'].mean()
        sent_conf_mean = df['sentence_confidence'].mean()
        word_conf_std_mean = df['word_confidence_std'].mean()
        
        # Per-class analysis
        per_class = df.groupby('true_label').agg({
            'word_correct': 'mean',
            'sentence_correct': 'mean',
            'word_aggregated_confidence': 'mean',
            'sentence_confidence': 'mean'
        }).to_dict()
        
        analysis[feature_type] = {
            'n_samples': len(df),
            'word_accuracy': float(word_acc),
            'sentence_accuracy': float(sent_acc),
            'agreement_rate': float(agreement_rate),
            'word_confidence_mean': float(word_conf_mean),
            'sentence_confidence_mean': float(sent_conf_mean),
            'word_confidence_std_mean': float(word_conf_std_mean),
            'per_class': per_class
        }
        
        print(f"\n  {feature_type} Results:")
        print(f"    Word-level accuracy: {word_acc:.4f}")
        print(f"    Sentence-level accuracy: {sent_acc:.4f}")
        print(f"    Agreement rate: {agreement_rate:.4f}")
        print(f"    Word avg confidence: {word_conf_mean:.4f} (±{word_conf_std_mean:.4f})")
        print(f"    Sentence avg confidence: {sent_conf_mean:.4f}")
    
    return analysis

# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(results_mfcc, results_hubert):
    """Create comprehensive comparison plots"""
    print("\n[4/6] Generating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    df_mfcc = pd.DataFrame(results_mfcc) if results_mfcc else None
    df_hubert = pd.DataFrame(results_hubert) if results_hubert else None
    
    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = []
    mfcc_vals = []
    hubert_vals = []
    
    if df_mfcc is not None:
        metrics.extend(['Word\nMFCC', 'Sentence\nMFCC'])
        mfcc_vals.extend([df_mfcc['word_correct'].mean(), df_mfcc['sentence_correct'].mean()])
    
    if df_hubert is not None:
        metrics.extend(['Word\nHuBERT', 'Sentence\nHuBERT'])
        hubert_vals.extend([df_hubert['word_correct'].mean(), df_hubert['sentence_correct'].mean()])
    
    x_pos = np.arange(len(metrics))
    colors = ['#1f77b4', '#ff7f0e'] * 2
    ax1.bar(x_pos, mfcc_vals + hubert_vals, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Word vs Sentence Accuracy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Confidence comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if df_mfcc is not None and df_hubert is not None:
        data_conf = [
            df_mfcc['word_aggregated_confidence'],
            df_mfcc['sentence_confidence'],
            df_hubert['word_aggregated_confidence'],
            df_hubert['sentence_confidence']
        ]
        ax2.boxplot(data_conf, labels=['Word\nMFCC', 'Sent\nMFCC', 'Word\nHuBERT', 'Sent\nHuBERT'])
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Agreement rate
    ax3 = fig.add_subplot(gs[0, 2])
    if df_mfcc is not None and df_hubert is not None:
        agreement = [df_mfcc['agreement'].mean(), df_hubert['agreement'].mean()]
        ax3.bar(['MFCC', 'HuBERT'], agreement, color=['#1f77b4', '#2ca02c'])
        ax3.set_ylabel('Agreement Rate')
        ax3.set_title('Word-Sentence Agreement')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-5. Per-class accuracy (MFCC and HuBERT)
    for idx, (df, title, ax_pos) in enumerate([
        (df_mfcc, 'MFCC Per-Class Accuracy', gs[1, 0]),
        (df_hubert, 'HuBERT Per-Class Accuracy', gs[1, 1])
    ]):
        if df is not None:
            ax = fig.add_subplot(ax_pos)
            per_class = df.groupby('true_label').agg({
                'word_correct': 'mean',
                'sentence_correct': 'mean'
            })
            per_class.plot(kind='bar', ax=ax, rot=45)
            ax.set_title(title)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Class')
            ax.legend(['Word-level', 'Sentence-level'])
            ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Stability analysis (confidence std vs n_words)
    ax6 = fig.add_subplot(gs[1, 2])
    if df_hubert is not None:
        ax6.scatter(df_hubert['n_words'], df_hubert['word_confidence_std'], alpha=0.5)
        ax6.set_xlabel('Number of Words')
        ax6.set_ylabel('Confidence Std Dev')
        ax6.set_title('Prediction Stability (HuBERT)')
        ax6.grid(True, alpha=0.3)
    
    # 7-8. Confusion matrices
    for idx, (df, title, ax_pos) in enumerate([
        (df_mfcc, 'MFCC Word-Level', gs[2, 0]),
        (df_hubert, 'HuBERT Word-Level', gs[2, 1])
    ]):
        if df is not None:
            ax = fig.add_subplot(ax_pos)
            labels = sorted(df['true_label'].unique())
            cm = confusion_matrix(df['true_label'], df['word_aggregated_label'], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                       yticklabels=labels, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
    
    # 9. Summary table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = "Word vs Sentence Comparison\n" + "="*35 + "\n\n"
    
    for feature_type, df in [('MFCC', df_mfcc), ('HuBERT', df_hubert)]:
        if df is not None:
            summary_text += f"{feature_type}:\n"
            summary_text += f"  Word Acc: {df['word_correct'].mean():.4f}\n"
            summary_text += f"  Sent Acc: {df['sentence_correct'].mean():.4f}\n"
            summary_text += f"  Agreement: {df['agreement'].mean():.4f}\n"
            summary_text += f"  Samples: {len(df)}\n\n"
    
    ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Word-Level vs Sentence-Level Accent Detection', fontsize=16, fontweight='bold')
    plt.savefig('results/word_sentence_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/word_sentence_comparison.png")

# ============================================================================
# Save Results
# ============================================================================

def save_results(results_mfcc, results_hubert, analysis):
    """Save detailed results to JSON"""
    print("\n[5/6] Saving results...")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'sample_rate': SAMPLE_RATE,
            'min_word_duration': MIN_WORD_DURATION,
            'max_word_duration': MAX_WORD_DURATION,
            'silence_threshold': SILENCE_THRESHOLD,
            'hubert_layer': BEST_HUBERT_LAYER
        },
        'analysis': analysis,
        'mfcc_results': results_mfcc[:100] if results_mfcc else [],  # Save sample
        'hubert_results': results_hubert[:100] if results_hubert else []
    }
    
    with open('results/word_sentence_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("  ✓ Saved: results/word_sentence_comparison.json")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Word vs Sentence level accent detection')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory with audio files')
    parser.add_argument('--max-files', type=int, help='Limit number of files to process')
    parser.add_argument('--skip-mfcc', action='store_true', help='Skip MFCC analysis')
    parser.add_argument('--skip-hubert', action='store_true', help='Skip HuBERT analysis')
    args = parser.parse_args()
    
    # Load models
    models = load_models()
    
    results_mfcc = []
    results_hubert = []
    
    # Process with MFCC
    if not args.skip_mfcc and models.get('mfcc_model') is not None:
        results_mfcc = process_dataset(args.data_dir, models, 'mfcc', args.max_files)
    
    # Process with HuBERT
    if not args.skip_hubert and models.get('hubert_model') is not None:
        results_hubert = process_dataset(args.data_dir, models, 'hubert', args.max_files)
    
    # Analyze
    analysis = analyze_results(results_mfcc, results_hubert)
    
    # Visualize
    plot_comparison(results_mfcc, results_hubert)
    
    # Save
    save_results(results_mfcc, results_hubert, analysis)
    
    print("\n[6/6] Generating detailed report...")
    
    print("\n" + "="*80)
    print("✅ Word vs Sentence comparison complete!")
    print("="*80)
    
    # Print key findings
    if results_mfcc:
        df_mfcc = pd.DataFrame(results_mfcc)
        print(f"\nMFCC: Word-level ({df_mfcc['word_correct'].mean():.4f}) vs " +
              f"Sentence-level ({df_mfcc['sentence_correct'].mean():.4f})")
    
    if results_hubert:
        df_hubert = pd.DataFrame(results_hubert)
        print(f"HuBERT: Word-level ({df_hubert['word_correct'].mean():.4f}) vs " +
              f"Sentence-level ({df_hubert['sentence_correct'].mean():.4f})")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
