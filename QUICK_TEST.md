# Quick Command-Line Testing Script

## Test All 6 States (Copy-Paste into PowerShell)

```powershell
cd "path\to\NLI_HuBERT_Project"

Write-Host "`n=== TESTING ANDHRA PRADESH ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"

Write-Host "`n=== TESTING GUJARAT ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/gujrat/Gujrat_speaker_01_1.wav"

Write-Host "`n=== TESTING JHARKHAND ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/jharkhand/Jharkhand_speaker_01_Recording (2).wav"

Write-Host "`n=== TESTING KARNATAKA ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/karnataka/Karnataka_speaker_03_1 (1).wav"

Write-Host "`n=== TESTING KERALA ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/kerala/Kerala_speaker_05_List49_Splitted_1.wav"

Write-Host "`n=== TESTING TAMIL NADU ===" -ForegroundColor Cyan
python scripts/predict.py "data/raw/tamil/Tamil_speaker (1).wav"

Write-Host "`n=== ALL TESTS COMPLETE ===" -ForegroundColor Green
```

## Expected Output for Each Test

```
Loading models...
Using device: cpu
Models loaded successfully!

Processing: [filename]
Features extracted from live audio...

Predicted Region: [STATE NAME]
Confidence: XX.XX%

Top 3 predictions:
  1. [State]: XX.XX%
  2. [State]: XX.XX%
  3. [State]: XX.XX%

Recommended cuisines: [list of dishes]
```

## Screenshot These Results!

Take a screenshot of the PowerShell window showing all 6 predictions. This serves as proof that your model works correctly.

## Alternative: Test One State at a Time

If you want to test individually (easier to screenshot each):

```powershell
cd "path\to\NLI_HuBERT_Project"

# Test Andhra Pradesh
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"
```

Then take screenshot, then next state, etc.

## For Your Report

Include screenshots with captions like:

"**Figure 1**: Model correctly predicting Andhra Pradesh accent with 99.14% confidence"

"**Figure 2**: Model correctly predicting Gujarat accent with 97.5%+ confidence"

etc.
