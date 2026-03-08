# Native Language Identification using HuBERT

Native Language Identification (NLI) system for classifying Indian regional accents from English speech.

This project combines:
- HuBERT embeddings (layer-wise analysis, best at Layer 3)
- MFCC baseline features
- Classical ML and deep learning experiments
- A Gradio web app for accent prediction and cuisine recommendation

## Highlights

- Final model: HuBERT Layer 3 + Random Forest
- Reported test accuracy: up to 99.73% on the adult test split
- Regions: Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu
- Includes: layer-wise analysis, word-vs-sentence study, and adult-to-child generalization checks

## Repository Structure

```text
.
|-- app.py
|-- requirements.txt
|-- scripts/
|-- data/
|   `-- manifests/
|-- features/
|-- models/
|-- results/
|-- reports/
|-- PROJECT_OVERVIEW.md
|-- PROJECT_REPORT.md
`-- TESTING_GUIDE.md
```

Notes:
- Large audio/features/model artifacts are excluded from Git via `.gitignore`.
- Folder placeholders are kept with `.gitkeep` where needed.

## Setup

### 1) Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Predict from a single audio file

```bash
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"
```

### Run web app

```bash
python app.py
```

Then open the local Gradio URL shown in terminal.

## Reproducible Workflow

### 1) Extract features

```bash
python scripts/batch_extract_features.py
```

### 2) Train final model

```bash
python scripts/train_final_model.py
```

### 3) Evaluate / predict

```bash
python scripts/predict.py <path_to_audio.wav>
```

## Documentation

- `PROJECT_OVERVIEW.md`: full technical summary and results
- `PROJECT_REPORT.md`: project report
- `TESTING_GUIDE.md`: validation and testing steps
- `QUICK_TEST.md`: quick command-line checks
- `SUBMISSION_READY.md`: submission checklist

## Limitations

- Performance can drop for out-of-distribution speakers and recording conditions.
- Child speech generalization is limited in current setup.
- Data augmentation and domain adaptation are proposed as future improvements.

## License

This project is licensed under the MIT License. See `LICENSE`.
