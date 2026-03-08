import numpy as np
import joblib

print("Loading models...")
model = joblib.load('models/rf_hubert_final.joblib')
scaler = joblib.load('models/scaler_hubert.joblib')
pca = joblib.load('models/pca_hubert.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')
print("  - Loaded: Model, Scaler, PCA, Label Encoder\n")

# Test speaker 1084
feature_path = "features/hubert/Andhra_speaker (1084).npz"
print(f"Testing: Andhra_speaker (1084)")
print(f"Ground truth: andhra_pradesh\n")

# Load features (same as training)
data = np.load(feature_path)
pooled = data['pooled']  # Shape: (26 layers, 768)
features = pooled[3]  # Extract Layer 3 (index 3) - shape: (768,)

# Preprocess
features_scaled = scaler.transform(features.reshape(1, -1))
features_pca = pca.transform(features_scaled)

# Predict
prediction = model.predict(features_pca)[0]
probabilities = model.predict_proba(features_pca)[0]
predicted_region = label_encoder.inverse_transform([prediction])[0]
confidence = probabilities[prediction] * 100

# Get top 3
top_indices = np.argsort(probabilities)[::-1][:3]
top_regions = label_encoder.inverse_transform(top_indices)
top_confidences = probabilities[top_indices] * 100

print(f"{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Predicted: {predicted_region}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nTop 3 Predictions:")
for i, (region, conf) in enumerate(zip(top_regions, top_confidences), 1):
    print(f"  {i}. {region}: {conf:.2f}%")
print(f"{'='*60}")

# Ground truth check
is_correct = predicted_region == 'andhra_pradesh'
print(f"✓ CORRECT!" if is_correct else f"✗ INCORRECT (should be andhra_pradesh)")
