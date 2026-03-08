# Native Language Identification (NLI) using HuBERT and MFCC

An end-to-end speech ML project that predicts a speaker's native Indian region from English accent patterns.

The repository includes feature extraction pipelines, training scripts, evaluation utilities, and a Gradio app for interactive inference.

## Project Summary

This project explores accent identification using two feature families:
- HuBERT embeddings (layer-wise analysis across layers 0-12)
- MFCC features (baseline)

It evaluates multiple modeling approaches and deployment-facing behavior:
- Layer-wise HuBERT performance comparison
- Word-level vs sentence-level input comparison
- Adult-to-child generalization behavior
- Interactive web demo with accent-aware cuisine recommendation

## Key Results

- Best configuration: HuBERT Layer 3 + Random Forest
- Reported accuracy: up to 99.73% on the adult speaker-independent test split
- Regions/classes:
	- Andhra Pradesh
	- Gujarat
	- Jharkhand
	- Karnataka
	- Kerala
	- Tamil Nadu

## Tech Stack

- Python
- PyTorch + Torchaudio
- Hugging Face Transformers (HuBERT)
- scikit-learn
- Librosa
- Gradio

## Repository Layout

```text
NLI_HuBERT_Project/
	app.py
	requirements.txt
	README.md
	PROJECT_OVERVIEW.md
	PROJECT_REPORT.md
	TESTING_GUIDE.md
	QUICK_TEST.md
	SUBMISSION_READY.md
	scripts/
	data/
		manifests/
	features/
	models/
	results/
	reports/
```

Notes:
- Large generated artifacts (audio/features/model binaries/results) are excluded via `.gitignore`.
- Directory placeholders are kept using `.gitkeep` files.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run a single prediction

```bash
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"
```

### Launch the web app

```bash
python app.py
```

Open the local URL printed by Gradio in terminal.

## Training and Evaluation Workflow

### 1. Feature extraction

```bash
python scripts/batch_extract_features.py
```

### 2. Train final model

```bash
python scripts/train_final_model.py
```

### 3. Predict on new audio

```bash
python scripts/predict.py <path_to_audio.wav>
```

## Experiments Included

- HuBERT layer-wise analysis
- MFCC baseline vs HuBERT comparison
- Word vs sentence-level robustness study
- Adult-to-child transfer/generalization checks

## Known Limitations

- Generalization can drop for unseen speakers, microphones, and environments.
- Child speech performance is weaker than adult-speech performance.
- Synthetic/TTS audio may be flagged as uncertain or out-of-distribution.

## Documentation

- `PROJECT_OVERVIEW.md`: end-to-end technical overview and analysis
- `PROJECT_REPORT.md`: project report
- `TESTING_GUIDE.md`: verification checklist and test flow
- `QUICK_TEST.md`: quick CLI test commands
- `SUBMISSION_READY.md`: submission-oriented checklist

## Reproducibility Notes

- Use the same dependency versions from `requirements.txt`.
- Keep random seeds fixed where configured in scripts.
- Ensure consistent audio preprocessing settings across train/test/inference.

## Author

- Gagan Reddy

## License

This project is licensed under the MIT License.
See `LICENSE` for full text.
This project is licensed under the MIT License. See `LICENSE`.
