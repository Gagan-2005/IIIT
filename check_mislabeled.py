"""Identify all mislabeled samples in the dataset"""
import pandas as pd
from pathlib import Path

# Load predictions from training
pred_file = Path("results/rf_final_predictions.csv")
if pred_file.exists():
    df = pd.read_csv(pred_file)
    wrong = df[df['correct'] == False]
    
    print("="*70)
    print(f"MISCLASSIFIED SAMPLES: {len(wrong)}/{len(df)} ({len(wrong)/len(df)*100:.2f}%)")
    print("="*70)
    
    if len(wrong) > 0:
        print("\nConfusion patterns:")
        print(wrong[['true_label', 'predicted_label']].value_counts())
        
        # Check if speaker 1083 is in test set
        metadata = pd.read_csv("metadata_existing.csv")
        speaker_1083 = metadata[metadata['wav_path'].str.contains('1083', na=False)]
        print("\n" + "="*70)
        print("Speaker 1083 info:")
        print(speaker_1083[['wav_path', 'label']])
    else:
        print("\n✅ Perfect test set accuracy!")
else:
    print("⚠️  Prediction results file not found. Model may not have been evaluated yet.")
    print("Running quick evaluation...")
    
    # Quick check on metadata
    import sys
    sys.path.insert(0, 'scripts')
    from predict import load_models, predict_native_language
    
    metadata = pd.read_csv("metadata_existing.csv")
    model, scaler, pca, le = load_models()
    
    # Test speaker 1083 and 1084
    test_speakers = ['1083', '1084', '1085']
    
    for spk in test_speakers:
        rows = metadata[metadata['wav_path'].str.contains(f'Andhra_speaker \\({spk}\\)', na=False, regex=True)]
        if len(rows) > 0:
            row = rows.iloc[0]
            audio_path = row['wav_path']
            true_label = row['label']
            
            if Path(audio_path).exists():
                result = predict_native_language(audio_path, model, scaler, pca, le)
                pred = result['predicted_label']
                conf = result['confidence']
                status = "✓" if pred == true_label else "✗"
                
                print(f"{status} Speaker {spk}: True={true_label:15s} Pred={pred:15s} Conf={conf*100:5.1f}%")
