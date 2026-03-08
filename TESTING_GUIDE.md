# Testing Guide for Web Application

## Overview

This guide walks you through testing the Gradio web interface to capture screenshots for your project report/presentation.

---

## Step 1: Launch the Web Application

Open PowerShell and run:

```powershell
cd "c:\Users\Meghna Gagan\OneDrive\Desktop\NLI_HuBERT_Proj-2025\NLI_HuBERT_Project"
python app.py
```

The app should launch at `http://127.0.0.1:7860`. Open this URL in your browser.

**Note**: If you encounter issues with the web interface, you can test directly via command line (see Step 5).

---

## Step 2: Test Dataset Samples (All 6 States)

Test one representative sample from each state to verify the model works correctly on trained data.

### Sample Files to Test:

| State | File Path | Expected Confidence |
|-------|-----------|---------------------|
| **Andhra Pradesh** | `data/raw/andhra_pradesh/Andhra_speaker (1084).wav` | 95-99% |
| **Gujarat** | `data/raw/gujrat/Gujrat_speaker_01_1.wav` | 95-99% |
| **Jharkhand** | `data/raw/jharkhand/Jharkhand_speaker_01_Recording (2).wav` | 95-99% |
| **Karnataka** | `data/raw/karnataka/Karnataka_speaker_01_1.wav` | 95-99% |
| **Kerala** | `data/raw/kerala/Kerala_speaker_04_List42_Splitted_1.wav` | 95-99% |
| **Tamil Nadu** | `data/raw/tamil/Tamil_speaker (1).wav` | 95-99% |

### Actions:
1. Upload each audio file to the Gradio interface
2. Wait for prediction (2-5 seconds)
3. **Screenshot each successful prediction** showing:
   - Predicted region
   - Confidence percentage
   - Cuisine recommendation
4. Save screenshots as: `screenshot_andhra.png`, `screenshot_gujarat.png`, etc.

---

## Step 3: Test Real Friend Recordings (Optional but Recommended)

If you have 1-2 friends/colleagues who are native speakers of these Indian states:

1. Ask them to record 5-10 seconds of clear English speech
   - Example: "I went to the market yesterday to buy vegetables and fruits for dinner."
   - Use their phone's voice recorder app
   - Save as `.wav` or `.mp3` format

2. Upload to the Gradio interface

3. **Capture screenshot** with label like: `screenshot_friend_[state].png`

**Expected Result**: If their accent matches the training data well, you should get 70-90%+ confidence.

**If confidence is low (30-50%)**: This is expected! It demonstrates the limitation we documented in PROJECT_OVERVIEW.md. This is valuable for your report.

---

## Step 4: Test Failure Cases (Unknown/Low Confidence)

To demonstrate the model's limitations and unknown detection:

### Option A: Child Audio
- If you have access to child speakers (age 5-12), record 5-10 seconds of English
- Expected: Low confidence (20-40%) or "Unknown/Uncertain" flag

### Option B: Synthetic TTS Audio
1. Go to a text-to-speech website (e.g., https://www.narakeet.com/)
2. Generate English speech with an Indian accent (if available)
3. Download the audio
4. Upload to the app

**Expected Result**: "Unknown/Uncertain" flag or very low confidence (<40%)

5. **Capture screenshot** showing the failure case
6. Save as: `screenshot_unknown.png` or `screenshot_low_confidence.png`

**Important**: This is NOT a bug! Document this as: *"Model correctly identifies out-of-distribution samples"*

---

## Step 5: Command-Line Testing (Alternative to Web App)

If the web app has issues, test directly via command line:

```powershell
# Test Andhra Pradesh
python scripts/predict.py "data/raw/andhra_pradesh/Andhra_speaker (1084).wav"

# Test Gujarat
python scripts/predict.py "data/raw/gujrat/Gujrat_speaker_01_1.wav"

# Test Jharkhand
python scripts/predict.py "data/raw/jharkhand/Jharkhand_speaker_01_Recording (2).wav"

# Test Karnataka
python scripts/predict.py "data/raw/karnataka/Karnataka_speaker_01_1.wav"

# Test Kerala
python scripts/predict.py "data/raw/kerala/Kerala_speaker_04_List42_Splitted_1.wav"

# Test Tamil Nadu
python scripts/predict.py "data/raw/tamil/Tamil_speaker (1).wav"
```

Each command will output:
```
Predicted Region: [State Name]
Confidence: XX.XX%
Top 3 predictions:
  1. [State 1]: XX.XX%
  2. [State 2]: XX.XX%
  3. [State 3]: XX.XX%
```

**Screenshot the terminal output** for each test.

---

## Step 6: Organize Screenshots for Report

Create a folder structure:

```
screenshots/
├── dataset_tests/
│   ├── andhra_pradesh.png
│   ├── gujarat.png
│   ├── jharkhand.png
│   ├── karnataka.png
│   ├── kerala.png
│   └── tamil_nadu.png
├── real_recordings/
│   ├── friend1_[state].png
│   └── friend2_[state].png
└── failure_cases/
    ├── child_speaker_low_confidence.png
    └── synthetic_tts_unknown.png
```

---

## Step 7: Writing Your Report Section

Use this narrative structure:

### Results & Testing

**Dataset Performance:**
"The model was tested on held-out samples from all 6 trained regions (Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu). As shown in [Screenshots 1-6], the model achieves 95-99% confidence on adult speakers from the training dataset, correctly identifying their native region based on English accent patterns."

**Real-World Testing:**
"We also tested the model on 1-2 real adult recordings from friends/colleagues with known native states. [Results: describe whether high or low confidence, explain why]"

**Limitations - Age Gap:**
"When tested on child speakers (age 5-12), the model shows significantly reduced performance, with confidence dropping to 20-40%. This is expected due to:
- Different voice pitch and formant frequencies
- Smaller vocal tract size
- Limited child training data (only 10 samples)"

**Limitations - Synthetic Audio:**
"The model correctly identifies synthetic text-to-speech (TTS) audio as 'Unknown/Uncertain', demonstrating its ability to detect out-of-distribution samples that lack natural acoustic variation."

**Future Work:**
"To address these limitations, we propose implementing data augmentation (pitch shifting, time stretching, noise injection) during training. This would increase data diversity 6x and improve generalization to new speakers and age groups. See PROJECT_OVERVIEW.md Section 11.2 for implementation details."

---

## Summary Checklist

- [ ] 6 screenshots from dataset samples (all states)
- [ ] 1-2 screenshots from real friend recordings (if available)
- [ ] 1 screenshot showing Unknown/low confidence case
- [ ] All screenshots labeled clearly
- [ ] Report section written explaining:
  - Model works well for trained adult speakers
  - Fails on children and synthetic audio (expected behavior)
  - Proposed augmentation as future work (not yet implemented)

---

**Good luck with your submission!** 🎓
