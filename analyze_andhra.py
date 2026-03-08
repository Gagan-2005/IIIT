"""Analyze Andhra predictions from training"""
import pandas as pd

# Load predictions from training
preds = pd.read_csv('results/rf_speaker_normalized_predictions.csv')

# Filter Andhra test samples
andhra_test = preds[preds['true_label'] == 'andhra_pradesh']

print("="*70)
print("ANDHRA PRADESH TEST SET ANALYSIS")
print("="*70)

print(f"\nTotal Andhra test samples: {len(andhra_test)}")
print(f"\nPrediction breakdown:")
print(andhra_test['predicted_label'].value_counts())

accuracy = (andhra_test['true_label'] == andhra_test['predicted_label']).mean()
print(f"\nAndhra Test Accuracy: {accuracy*100:.1f}%")

# Show misclassified samples
misclassified = andhra_test[andhra_test['true_label'] != andhra_test['predicted_label']]
if len(misclassified) > 0:
    print(f"\nMisclassified as:")
    print(misclassified['predicted_label'].value_counts())
    print(f"\nSample files misclassified:")
    for idx, row in misclassified.head(5).iterrows():
        print(f"  - {row['sample_id']}: predicted as {row['predicted_label']}")
