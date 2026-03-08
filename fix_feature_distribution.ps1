# ============================================================================
# PERMANENT FIX: Feature Distribution Mismatch
# ============================================================================
# This script re-extracts all features with consistent code and retrains
# Estimated time: 2-3 hours for extraction + 10 minutes for training
# ============================================================================

Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  FEATURE DISTRIBUTION FIX - AUTOMATED RECOVERY" -ForegroundColor Yellow
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# ============================================================================
# STEP 0: Pre-flight checks
# ============================================================================
Write-Host "[0/4] Pre-flight checks..." -ForegroundColor Cyan

# Check if Python is available
$PythonCheck = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Python found" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Python not found!" -ForegroundColor Red
    exit 1
}

# Check required files
$RequiredFiles = @(
    "scripts\batch_extract_features.py",
    "scripts\train_final_model.py",
    "metadata.csv"
)

foreach ($file in $RequiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] Found: $file" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Missing: $file" -ForegroundColor Red
        exit 1
    }
}

# Check current feature count
$CurrentFeatures = (Get-ChildItem "features\hubert\*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "  [INFO] Current features: $CurrentFeatures" -ForegroundColor Yellow

Write-Host ""
Write-Host "This script will:" -ForegroundColor White
Write-Host "  1. Backup current model" -ForegroundColor White
Write-Host "  2. Delete all cached features" -ForegroundColor White
Write-Host "  3. Re-extract 3,701 features (~2-3 hours)" -ForegroundColor White
Write-Host "  4. Retrain model (~10 minutes)" -ForegroundColor White
Write-Host "  5. Validate predictions" -ForegroundColor White
Write-Host ""
Write-Host "WARNING: This will take 2-3 hours. Keep this window open!" -ForegroundColor Red
Write-Host ""

$Confirm = Read-Host "Continue? (yes/no)"
if ($Confirm -ne "yes") {
    Write-Host "Aborted by user." -ForegroundColor Yellow
    exit 0
}

# ============================================================================
# STEP 1: Backup current model
# ============================================================================
Write-Host ""
Write-Host "[1/4] Backing up current model..." -ForegroundColor Cyan

$BackupSuffix = (Get-Date -Format "yyyyMMdd_HHmmss")
$ModelsToBackup = @(
    "models\rf_hubert_final.joblib",
    "models\scaler_hubert.joblib",
    "models\pca_hubert.joblib",
    "models\label_encoder.joblib"
)

foreach ($model in $ModelsToBackup) {
    if (Test-Path $model) {
        $BackupPath = $model -replace "\.joblib$", "_backup_$BackupSuffix.joblib"
        Copy-Item $model $BackupPath
        Write-Host "  [OK] Backed up: $model" -ForegroundColor Green
    }
}

# ============================================================================
# STEP 2: Clear all cached features
# ============================================================================
Write-Host ""
Write-Host "[2/4] Clearing cached features..." -ForegroundColor Cyan

$FeaturePath = "features\hubert"
if (Test-Path $FeaturePath) {
    $OldCount = (Get-ChildItem "$FeaturePath\*.npz" | Measure-Object).Count
    Write-Host "  [INFO] Removing $OldCount feature files..." -ForegroundColor Yellow
    Remove-Item "$FeaturePath\*.npz" -Force
    Write-Host "  [OK] Cleared all features" -ForegroundColor Green
} else {
    Write-Host "  [INFO] No features to clear" -ForegroundColor Yellow
}

# ============================================================================
# STEP 3: Re-extract all features
# ============================================================================
Write-Host ""
Write-Host "[3/4] Re-extracting features (this will take 2-3 hours)..." -ForegroundColor Cyan
Write-Host "  [INFO] Progress will be shown below" -ForegroundColor Yellow
Write-Host "  [WARNING] DO NOT CLOSE THIS WINDOW" -ForegroundColor Red
Write-Host ""

$ExtractionStart = Get-Date

python scripts\batch_extract_features.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [ERROR] Feature extraction failed!" -ForegroundColor Red
    exit 1
}

$ExtractionEnd = Get-Date
$ExtractionTime = ($ExtractionEnd - $ExtractionStart).TotalMinutes
Write-Host ""
Write-Host "  [OK] Feature extraction complete!" -ForegroundColor Green
Write-Host "  [INFO] Time taken: $([math]::Round($ExtractionTime, 1)) minutes" -ForegroundColor Yellow

# Verify feature count
$NewFeatures = (Get-ChildItem "features\hubert\*.npz" | Measure-Object).Count
Write-Host "  [INFO] Extracted features: $NewFeatures" -ForegroundColor Yellow

if ($NewFeatures -lt 3500) {
    Write-Host "  [WARNING] Expected ~3,701 features, got $NewFeatures" -ForegroundColor Yellow
    $Continue = Read-Host "Continue anyway? (yes/no)"
    if ($Continue -ne "yes") {
        Write-Host "Aborted by user." -ForegroundColor Yellow
        exit 0
    }
}

# ============================================================================
# STEP 4: Retrain model
# ============================================================================
Write-Host ""
Write-Host "[4/4] Retraining model..." -ForegroundColor Cyan
Write-Host ""

$TrainingStart = Get-Date

python scripts\train_final_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [ERROR] Model training failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  To restore backup:" -ForegroundColor Yellow
    Write-Host "    Copy-Item models\rf_hubert_final_backup_$BackupSuffix.joblib models\rf_hubert_final.joblib" -ForegroundColor White
    exit 1
}

$TrainingEnd = Get-Date
$TrainingTime = ($TrainingEnd - $TrainingStart).TotalMinutes
Write-Host ""
Write-Host "  [OK] Model training complete!" -ForegroundColor Green
Write-Host "  [INFO] Time taken: $([math]::Round($TrainingTime, 1)) minutes" -ForegroundColor Yellow

# ============================================================================
# STEP 5: Validation
# ============================================================================
Write-Host ""
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  VALIDATION" -ForegroundColor Yellow
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "Testing predictions..." -ForegroundColor Cyan
Write-Host ""

# Test with pre-extracted features
Write-Host "[Test 1] Pre-extracted features:" -ForegroundColor White
if (Test-Path "test_prediction.py") {
    python test_prediction.py
} else {
    Write-Host "  [WARNING] test_prediction.py not found, skipping" -ForegroundColor Yellow
}

Write-Host ""

# Test with real-time extraction
Write-Host "[Test 2] Real-time extraction:" -ForegroundColor White
if (Test-Path "test_realtime_extraction.py") {
    python test_realtime_extraction.py
} else {
    Write-Host "  [INFO] test_realtime_extraction.py not found, skipping" -ForegroundColor Yellow
}

# ============================================================================
# Summary
# ============================================================================
$EndTime = Get-Date
$TotalTime = ($EndTime - $StartTime).TotalMinutes

Write-Host ""
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "  RECOVERY COMPLETE" -ForegroundColor Green
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""
Write-Host "Total time: $([math]::Round($TotalTime, 1)) minutes" -ForegroundColor Yellow
Write-Host ""
Write-Host "[OK] Features re-extracted with consistent code" -ForegroundColor Green
Write-Host "[OK] Model retrained on new features" -ForegroundColor Green
Write-Host "[OK] Predictions should now work correctly for new audio files" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Test with your own audio: python scripts\predict.py your_audio.wav" -ForegroundColor White
Write-Host "  2. Run web app: python app.py" -ForegroundColor White
Write-Host "  3. Remove workarounds from app.py if tests pass" -ForegroundColor White
Write-Host ""
Write-Host "Backup location: models\*_backup_$BackupSuffix.joblib" -ForegroundColor Yellow
Write-Host ""
