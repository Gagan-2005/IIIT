![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# Native Language Identification of Indian English Speakers using HuBERT

A Deep Learning Project for Identifying Regional Indian Accents in English Speech

This project develops an intelligent system that identifies which Indian state someone is from by analyzing their English accent. Using Facebook's self-supervised HuBERT speech model combined with Random Forest classification, we achieve 99.73% accuracy in distinguishing between six major Indian regional accents.

## Project Overview

Native Language Identification (NLI) aims to predict a speaker's native language background from their speech patterns when speaking a second language (in this case, English). This has applications in personalized education, customer service, healthcare communication, and cultural technology solutions.

## Key Features

- High Accuracy Classification: 99.73% test accuracy using HuBERT Layer 3 embeddings
- Multi-Model Comparison: Evaluated MFCC baseline, HuBERT layers, CNN, BiLSTM, and Transformers
- Real-Time Web Application: Interactive Gradio interface for live accent detection
- Cuisine Recommendation System: Practical application suggesting regional dishes based on detected accent
- Comprehensive Analysis: Layer-wise HuBERT evaluation, cross-age generalization, word vs sentence-level comparison

## Supported Indian Regions

| Region | Language | Sample Dishes |
|--------|----------|---------------|
| Andhra Pradesh | Telugu | Hyderabadi Biryani, Gongura Pachadi |
| Gujarat | Gujarati | Dhokla, Thepla, Khandvi |
| Jharkhand | Hindi | Litti Chokha, Dhuska |
| Karnataka | Kannada | Bisi Bele Bath, Mysore Dosa |
| Kerala | Malayalam | Appam and Stew, Puttu and Kadala |
| Tamil Nadu | Tamil | Idli-Sambar, Chettinad Chicken |

## Performance Results

### Model Comparison

| Model | Features | Accuracy | F1-Score | Notes |
|-------|----------|----------|----------|-------|
| HuBERT Layer 3 + RF | 768->128 PCA | 99.73% | 99.69% | Best Model |
| MFCC Baseline + RF | 39-dim | 92.07% | 91.24% | Traditional features |
| HuBERT Layer 8 + RF | 768->128 PCA | 93.97% | 91.60% | Good but not optimal |
| HuBERT Multilayer + RF | All 13 layers | 96.13% | 64.96% | High dimensionality hurts |
| CNN | HuBERT L3 | 70.34% | 65.87% | Small dataset limitation |
| BiLSTM | HuBERT L3 | 72.41% | 68.23% | Underfitting issues |
| Transformer | HuBERT L3 | 68.97% | 63.45% | Too complex for data size |

### Per-Region Performance

All regions achieve greater than 99% precision and recall:
- Andhra Pradesh: 100% precision
- Kerala: 100% precision
- Tamil Nadu: 99.7% precision
- Gujarat: 99.5% precision
- Jharkhand: 99.8% precision
- Karnataka: 99.2% precision

## Key Research Findings

1. Layer 3 is Optimal for Accents: Mid-level HuBERT representations (Layer 3) capture phonetic and prosodic accent patterns better than raw acoustic (Layer 0-2) or semantic (Layer 9-12) features.
2. Self-Supervised Beats Hand-Crafted: HuBERT embeddings (99.73%) significantly outperform traditional MFCC features (92.07%) by 7.66 percentage points.
3. Classical ML Wins on Small Data: Random Forest outperforms deep learning models (CNN, BiLSTM, Transformer) when dataset size is moderate (3,701 samples).
4. Sentence Beats Word Level: Full sentences provide more stable accent cues than isolated words (99.73% vs 75-80% accuracy).
5. Cross-Age Challenge: Adult-trained models struggle with child speech (50% accuracy) due to vocal tract differences, highlighting the need for age-diverse training data.

## Quick Start

### Prerequisites

- Python: 3.12 (recommended) or 3.8+
- RAM: 8GB minimum (16GB recommended)
- Storage: around 4-5 GB for models and features
- GPU: Optional (CPU is sufficient for inference)

### Installation

```bash
# Navigate to project directory
cd path/to/NLI_HuBERT_Project

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, librosa, transformers; print('All dependencies installed')"
```

### Key Dependencies

- `torch` - PyTorch for neural networks
- `transformers` - Hugging Face HuBERT model
- `librosa` - Audio processing
- `scikit-learn` - Machine learning algorithms
- `gradio` - Web application interface
- `numpy`, `pandas` - Data manipulation
- `joblib` - Model persistence

## Project Structure

```text
NLI_HuBERT_Project/
|-- data/
|   `-- raw/              # Audio files (6 regions)
|-- features/
|   `-- hubert/           # Pre-extracted embeddings
|-- models/               # Trained classifiers
|   |-- rf_hubert_final.joblib
|   |-- scaler_hubert.joblib
|   |-- pca_hubert.joblib
|   `-- label_encoder.joblib
|-- scripts/
|   |-- batch_extract_features.py
|   |-- train_final_model.py
|   `-- predict.py
|-- reports/              # Analysis and documentation
|-- results/              # Experiment outputs
|-- app.py                # Gradio web interface
|-- requirements.txt
`-- README.md
```

## Usage

### Option 1: Command Line Prediction

```bash
# Predict from audio file
python scripts/predict.py "path/to/audio.wav"

# Example output:
# Predicted Region: Andhra Pradesh
# Confidence: 99.14%
# Recommended Dishes: Hyderabadi Biryani, Gongura Pachadi, Pesarattu
```

### Option 2: Web Application (Recommended)

```bash
# Launch Gradio interface
python app.py

# Open browser to: http://127.0.0.1:7860
```

Web app features:
- Record audio directly in browser
- Upload WAV/MP3 files
- View confidence scores for all 6 regions
- Get regional cuisine recommendations
- Automatic low-confidence warnings

### Option 3: Train Your Own Model

```bash
# Step 1: Extract HuBERT features
python scripts/batch_extract_features.py

# Step 2: Train classifier
python scripts/train_final_model.py

# Model will be saved to models/ directory
```

## Experiments Conducted

1. Layer-wise HuBERT Analysis
   - Tested all 13 HuBERT layers (0-12) to find optimal accent representation
   - Result: Layer 3 achieved 99.73% accuracy (best)
2. MFCC Baseline Comparison
   - MFCC: 92.07% accuracy
   - HuBERT: 99.73% accuracy (+7.66% improvement)
3. Deep Learning vs Classical ML
   - Compared CNN, BiLSTM, Transformer with Random Forest
   - Winner: Random Forest (99.73%)
4. Word vs Sentence Level
   - Sentences: 99.73% accuracy (stable predictions)
   - Words: 75-80% accuracy (highly variable)
5. Cross-Age Generalization
   - Adults: 99.73% accuracy
   - Children: around 50% accuracy

## Best Practices

For best predictions:
- Use clear audio with minimal background noise
- Use 5-15 seconds of continuous speech
- Use 16 kHz mono WAV format (auto-converted if different)
- Speak naturally in English with your native accent

Expected performance:
- 95-99% confidence: Adult speakers from trained regions
- 30-60% confidence: New speakers and different recording conditions
- Low confidence: Child speakers, synthetic TTS, unknown accents

## Dataset Information

### Adult Speech Dataset

- Total samples: 3,701 valid audio files
- Format: WAV, 16 kHz, mono
- Duration: 3-10 seconds per sample
- Regions: 6 Indian states
- Split: 80% train (~2,961) / 20% test (~740)
- Strategy: Speaker-independent stratified split

### Distribution by Region

| Region | Language | Training Samples | Test Samples |
|--------|----------|------------------|--------------|
| Andhra Pradesh | Telugu | ~167 | ~42 |
| Gujarat | Gujarati | ~238 | ~60 |
| Jharkhand | Hindi | ~483 | ~121 |
| Karnataka | Kannada | ~600 | ~150 |
| Kerala | Malayalam | ~200 | ~50 |
| Tamil Nadu | Tamil | ~1,273 | ~317 |

### Child Speech Subset (Generalization Study)

- Samples: 10 children (ages 6-12)
- Purpose: Zero-shot cross-age evaluation
- Result: around 50% accuracy (expected due to vocal tract differences)

## Technical Architecture

### Pipeline Overview

```text
Audio Input (WAV/MP3)
	↓
[1] Preprocessing (16kHz mono conversion)
	↓
[2] HuBERT Feature Extraction (Layer 3)
	-> 768-dimensional embeddings
	↓
[3] Statistical Calibration (mean/std alignment)
	↓
[4] StandardScaler Normalization
	↓
[5] PCA Dimensionality Reduction (768 -> 128)
	-> Retains 95.74% variance
	↓
[6] Random Forest Classification (300 trees)
	↓
[7] Confidence Thresholding and Verification
	-> Andhra/Jharkhand disambiguation
	↓
Output: Region + Confidence + Cuisine Recommendations
```

### Model Components

| Component | File | Purpose |
|-----------|------|---------|
| Main Classifier | `rf_hubert_final.joblib` | Random Forest classifier |
| Feature Scaler | `scaler_hubert.joblib` | StandardScaler normalization |
| Dimensionality Reducer | `pca_hubert.joblib` | PCA (768->128) |
| Label Encoder | `label_encoder.joblib` | Region name mapping |
| Calibration Stats | `live_calibration.npz` | Mean/std alignment for live predictions |
| Pairwise Verifier | `andhra_jharkhand_verifier.joblib` | Binary verifier for close pair |

## Troubleshooting

### Low Confidence Predictions (below 60%)

Possible causes:
- Poor audio quality (background noise, echoes)
- Speaker not using English language
- Audio too short (below 2 seconds)
- Speaker from untrained region
- Child speaker or synthetic voice

Solutions:
- Re-record with better audio quality
- Ensure speaker uses natural English speech
- Use 5-15 seconds of speech
- Review top 2-3 predictions for plausible alternatives

### Performance Issues

- First prediction may be slow (model loading)
- Subsequent predictions are usually faster
- Ensure minimum 8GB RAM for smoother runs

## Known Limitations

- Generalization drops for completely unseen speakers and recording setups
- Child speech performance is significantly lower than adult speech
- Synthetic audio is often flagged as uncertain or unknown

## Testing and Validation

### Quick Test (PowerShell)

```powershell
# Open PowerShell and navigate to project directory
cd "path\to\NLI_HuBERT_Project"

# Test all regions
Write-Host "=== TESTING ANDHRA PRADESH ==="
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"

Write-Host "=== TESTING GUJARAT ==="
python scripts/predict.py "data/raw/gujrat/Gujrat_speaker_01_1.wav"

Write-Host "=== TESTING JHARKHAND ==="
python scripts/predict.py "data/raw/jharkhand/Jharkhand_speaker_01_Recording (2).wav"

Write-Host "=== TESTING KARNATAKA ==="
python scripts/predict.py "data/raw/karnataka/Karnataka_speaker_03_1 (1).wav"

Write-Host "=== TESTING KERALA ==="
python scripts/predict.py "data/raw/kerala/Kerala_speaker_05_List49_Splitted_1.wav"

Write-Host "=== TESTING TAMIL NADU ==="
python scripts/predict.py "data/raw/tamil/Tamil_speaker (1).wav"

Write-Host "=== ALL TESTS COMPLETE ==="
```

## Future Work and Improvements

Planned enhancements:
- Data augmentation (pitch shift, time stretch, additive noise)
- Speaker normalization (i-vectors/x-vectors)
- Regional expansion beyond current 6 states
- Unknown accent detection thresholding
- Production APIs and mobile deployment
- Multi-task learning (accent + age + gender)


## Project Status

- Test Accuracy: 99.73%
- Last Updated: November 2025
- Version: 1.0.0

## Acknowledgments

- HuBERT model: Facebook AI Research
- Core libraries: Hugging Face, PyTorch, scikit-learn, librosa, Gradio
- Institution: Hyderabad Institute of Technology and Management


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
