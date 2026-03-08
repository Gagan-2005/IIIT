"""
Script to inspect and understand .joblib model files
"""

import joblib
import os
from pathlib import Path

MODEL_DIR = "models"

print("=" * 80)
print("Model Files Inspector")
print("=" * 80)

# Get all .joblib files
joblib_files = list(Path(MODEL_DIR).glob("*.joblib"))

if not joblib_files:
    print(f"\nNo .joblib files found in {MODEL_DIR}/")
else:
    print(f"\nFound {len(joblib_files)} .joblib files:\n")
    
    for model_path in sorted(joblib_files):
        print(f"\n{'='*80}")
        print(f"File: {model_path.name}")
        print(f"Size: {model_path.stat().st_size / 1024:.2f} KB")
        print("-" * 80)
        
        try:
            # Load the model
            obj = joblib.load(model_path)
            
            # Display type
            print(f"Type: {type(obj).__name__}")
            
            # Display content based on type
            if isinstance(obj, dict):
                print(f"Dictionary with {len(obj)} keys:")
                for key, value in obj.items():
                    print(f"  - {key}: {type(value).__name__}")
                    if hasattr(value, 'n_estimators'):
                        print(f"      n_estimators: {value.n_estimators}")
                    if hasattr(value, 'classes_'):
                        print(f"      classes: {value.classes_}")
            else:
                # Single object
                if hasattr(obj, 'n_estimators'):
                    print(f"  - n_estimators: {obj.n_estimators}")
                if hasattr(obj, 'n_features_in_'):
                    print(f"  - n_features_in: {obj.n_features_in_}")
                if hasattr(obj, 'classes_'):
                    print(f"  - classes: {obj.classes_}")
                if hasattr(obj, 'n_components_'):
                    print(f"  - n_components: {obj.n_components_}")
                if hasattr(obj, 'feature_names_in_'):
                    print(f"  - feature names available: Yes")
                    
        except Exception as e:
            print(f"Error loading file: {e}")

print("\n" + "=" * 80)
print("How to use these files:")
print("=" * 80)
print("""
import joblib
import numpy as np

# 1. Load a model
model = joblib.load('models/rf_hubert_final.joblib')

# 2. Load preprocessing objects
scaler = joblib.load('models/scaler_hubert.joblib')
pca = joblib.load('models/pca_hubert.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# 3. Make predictions on new data
# Assuming you have features X_new
X_scaled = scaler.transform(X_new)
X_pca = pca.transform(X_scaled)
predictions = model.predict(X_pca)
predicted_labels = label_encoder.inverse_transform(predictions)

# 4. Get prediction probabilities
probabilities = model.predict_proba(X_pca)
""")

print("\n" + "=" * 80)
