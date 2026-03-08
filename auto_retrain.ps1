# Auto-Retrain Script
# Run this after feature extraction completes

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "AUTOMATIC RETRAINING SCRIPT" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

# Check if extraction is complete
Write-Host "`n[1/5] Checking feature extraction status..." -ForegroundColor Yellow
$featureCount = (Get-ChildItem "features\hubert\*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "   Features found: $featureCount / 3701" -ForegroundColor Gray

if ($featureCount -lt 3700) {
    Write-Host "`n❌ ERROR: Feature extraction not complete!" -ForegroundColor Red
    Write-Host "   Expected: 3,701 features" -ForegroundColor Red
    Write-Host "   Found: $featureCount features" -ForegroundColor Red
    Write-Host "`n   Please wait for extraction to finish, then run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "   ✓ Feature extraction complete!" -ForegroundColor Green

# Backup old model
Write-Host "`n[2/5] Backing up current model..." -ForegroundColor Yellow
Copy-Item "models\rf_hubert_final.joblib" "models\rf_hubert_final_OLD.joblib" -Force
Copy-Item "models\scaler_hubert.joblib" "models\scaler_hubert_OLD.joblib" -Force
Copy-Item "models\pca_hubert.joblib" "models\pca_hubert_OLD.joblib" -Force
Write-Host "   ✓ Backup saved as *_OLD.joblib" -ForegroundColor Green

# Re-train model
Write-Host "`n[3/5] Retraining Random Forest model..." -ForegroundColor Yellow
Write-Host "   This will take 5-10 minutes..." -ForegroundColor Gray
python scripts\train_final_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ ERROR: Training failed!" -ForegroundColor Red
    Write-Host "   Restoring backup..." -ForegroundColor Yellow
    Copy-Item "models\rf_hubert_final_OLD.joblib" "models\rf_hubert_final.joblib" -Force
    exit 1
}
Write-Host "   ✓ Model retrained successfully!" -ForegroundColor Green

# Test predictions
Write-Host "`n[4/5] Testing predictions..." -ForegroundColor Yellow
python test_prediction.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️  Test had issues, check output above" -ForegroundColor Yellow
} else {
    Write-Host "   ✓ Predictions working!" -ForegroundColor Green
}

# Test real-time extraction
Write-Host "`n[5/5] Testing real-time extraction..." -ForegroundColor Yellow
python test_realtime_extraction.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ⚠️  Real-time test had issues" -ForegroundColor Yellow
} else {
    Write-Host "   ✓ Real-time extraction works!" -ForegroundColor Green
}

# Summary
Write-Host "`n" + "="*70 -ForegroundColor Green
Write-Host "✅ RETRAINING COMPLETE!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Update app.py to remove error messages (allow real-time extraction)" -ForegroundColor White
Write-Host "2. Test web app: python app.py" -ForegroundColor White
Write-Host "3. Upload NEW audio files - should now work correctly!" -ForegroundColor White

Write-Host "`nYour Andhra/Telangana audio should now predict correctly!" -ForegroundColor Green
