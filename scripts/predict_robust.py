"""
Robust Native Language Identification with handling for untrained classes
Uses similarity-based fallback for classes not seen during training
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Import feature extraction
try:
    from extract_hubert_features import extract_hubert_features
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_hubert_features import extract_hubert_features

# Configuration
MODELS_DIR = Path("models")
BEST_LAYER = 3
CUISINE_MAPPING_PATH = "cuisine_mapping.json"
CONFIDENCE_THRESHOLD = 0.3  # Below this, flag as uncertain

# All possible classes in dataset
ALL_CLASSES = [
    'andhra_pradesh',
    'gujrat', 
    'jharkhand',
    'karnataka',
    'kerala',
    'tamil'
]

class RobustNLIPredictor:
    """
    Robust predictor that can handle classes not seen during training
    """
    
    def __init__(self, model_path=None):
        """Initialize predictor with model path"""
        self.model = None
        self.scaler = None
        self.pca = None
        self.label_encoder = None
        self.trained_classes = None
        self.all_classes = ALL_CLASSES
        self.reference_features = {}  # Store reference features for similarity matching
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model and components"""
        print(f"Loading model from: {model_path}")
        
        # Try to load as dictionary first (bundled model)
        try:
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data['clf']
                self.scaler = model_data.get('scaler')
                self.pca = model_data.get('pca')
                self.label_encoder = model_data.get('label_encoder')
                
                # Get trained classes
                if self.label_encoder:
                    self.trained_classes = list(self.label_encoder.classes_)
                else:
                    self.trained_classes = []
                
                print(f"  ✓ Loaded bundled model")
                print(f"  ✓ Trained on: {self.trained_classes}")
            else:
                # Single model file
                self.model = model_data
                self.load_separate_components()
        
        except Exception as e:
            print(f"  ⚠ Error loading model: {e}")
            raise
    
    def load_separate_components(self):
        """Load scaler, PCA, and encoder separately"""
        scaler_path = MODELS_DIR / "scaler_hubert.joblib"
        pca_path = MODELS_DIR / "pca_hubert.joblib"
        le_path = MODELS_DIR / "label_encoder.joblib"
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Loaded scaler")
        
        if pca_path.exists():
            self.pca = joblib.load(pca_path)
            print(f"  ✓ Loaded PCA")
        
        if le_path.exists():
            self.label_encoder = joblib.load(le_path)
            self.trained_classes = list(self.label_encoder.classes_)
            print(f"  ✓ Loaded label encoder")
            print(f"  ✓ Trained on: {self.trained_classes}")
    
    def extract_features(self, audio_path, device='cpu'):
        """Extract HuBERT features from audio file"""
        print(f"\n{'='*60}")
        print(f"Extracting features: {Path(audio_path).name}")
        print(f"{'='*60}")
        
        features_dict = extract_hubert_features(audio_path, output_path=None, device=device)
        pooled_features = features_dict['pooled']
        
        if pooled_features.shape[0] <= BEST_LAYER:
            raise ValueError(f"Expected at least {BEST_LAYER+1} layers, got {pooled_features.shape[0]}")
        
        layer_features = pooled_features[BEST_LAYER]
        print(f"  ✓ Using Layer {BEST_LAYER}: shape {layer_features.shape}")
        
        return layer_features
    
    def preprocess_features(self, features):
        """Apply scaling and PCA to features"""
        features = features.reshape(1, -1)
        
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.pca:
            features = self.pca.transform(features)
        
        return features
    
    def predict_with_confidence(self, features):
        """
        Make prediction with confidence scores
        Returns prediction for all possible classes, not just trained ones
        """
        # Get standard model prediction
        prediction_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Map to trained class
        predicted_trained_class = self.trained_classes[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Create probability dict for trained classes
        trained_probs = {
            cls: float(prob) 
            for cls, prob in zip(self.trained_classes, probabilities)
        }
        
        return {
            'predicted_class': predicted_trained_class,
            'confidence': confidence,
            'trained_probabilities': trained_probs,
            'is_trained_class': True
        }
    
    def predict_all_classes(self, features_raw, features_processed):
        """
        Predict across ALL classes including untrained ones
        Uses similarity matching for untrained classes
        """
        # Get prediction on trained classes
        base_prediction = self.predict_with_confidence(features_processed)
        
        # Initialize results for all classes
        all_class_scores = {}
        
        # Add trained class probabilities
        for cls in self.trained_classes:
            all_class_scores[cls] = base_prediction['trained_probabilities'][cls]
        
        # For untrained classes, use similarity to closest trained class
        untrained_classes = [cls for cls in self.all_classes if cls not in self.trained_classes]
        
        if untrained_classes:
            # Map untrained classes to similar trained classes (simple heuristic)
            similarity_map = self._get_class_similarity_map()
            
            for untrained_cls in untrained_classes:
                # Find most similar trained class
                similar_cls = similarity_map.get(untrained_cls, self.trained_classes[0])
                # Assign reduced probability (penalty for being untrained)
                all_class_scores[untrained_cls] = all_class_scores[similar_cls] * 0.3
        
        # Normalize probabilities
        total = sum(all_class_scores.values())
        if total > 0:
            all_class_scores = {k: v/total for k, v in all_class_scores.items()}
        
        # Find best overall prediction
        best_class = max(all_class_scores.items(), key=lambda x: x[1])
        
        return {
            'predicted_class': best_class[0],
            'confidence': best_class[1],
            'all_probabilities': all_class_scores,
            'is_trained_class': best_class[0] in self.trained_classes,
            'trained_classes': self.trained_classes,
            'untrained_classes': untrained_classes
        }
    
    def _get_class_similarity_map(self):
        """
        Map untrained classes to similar trained classes
        Based on linguistic/geographic proximity
        """
        # Simple similarity mapping (can be improved with embeddings)
        similarity = {
            'karnataka': 'andhra_pradesh',  # Both South Indian Dravidian
            'tamil': 'kerala',               # Both South Indian Dravidian
            'kerala': 'tamil',
            'andhra_pradesh': 'karnataka',
            'gujrat': 'jharkhand',          # Both have diverse dialects
            'jharkhand': 'gujrat'
        }
        return similarity
    
    def predict(self, audio_path, device='cpu', use_all_classes=True):
        """
        Main prediction function
        
        Args:
            audio_path: Path to audio file
            device: 'cpu' or 'cuda'
            use_all_classes: If True, attempt to predict all classes including untrained
        
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features_raw = self.extract_features(audio_path, device)
        features_processed = self.preprocess_features(features_raw)
        
        # Make prediction
        if use_all_classes:
            result = self.predict_all_classes(features_raw, features_processed)
        else:
            result = self.predict_with_confidence(features_processed)
        
        # Add confidence warning
        if result['confidence'] < CONFIDENCE_THRESHOLD:
            result['warning'] = f"⚠️  Low confidence ({result['confidence']*100:.1f}%) - prediction may be unreliable"
        
        # Sort all probabilities
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        result['top_5'] = sorted_probs[:5]
        
        return result

def load_cuisine_mapping():
    """Load cuisine recommendations"""
    with open(CUISINE_MAPPING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_output(prediction_result, cuisine_info):
    """Format and display results"""
    print(f"\n{'='*70}")
    print("🎯 PREDICTION RESULTS")
    print(f"{'='*70}")
    
    # Main prediction
    pred_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    
    trained_marker = "✓ TRAINED" if prediction_result['is_trained_class'] else "⚠ UNTRAINED (estimated)"
    
    print(f"\n🗣️  PREDICTED LANGUAGE: {pred_class.upper().replace('_', ' ')}")
    print(f"📊 Confidence: {confidence*100:.2f}% [{trained_marker}]")
    
    # Warning if low confidence
    if 'warning' in prediction_result:
        print(f"\n{prediction_result['warning']}")
    
    # Top 5 predictions
    print(f"\n📊 TOP 5 PREDICTIONS:")
    for i, (label, prob) in enumerate(prediction_result['top_5'], 1):
        trained = "✓" if label in prediction_result['trained_classes'] else "⚠"
        bar_length = int(prob * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"   {i}. {trained} {label:20s} {bar} {prob*100:.2f}%")
    
    # Training info
    print(f"\n📚 MODEL TRAINING INFO:")
    print(f"   Trained on: {', '.join(prediction_result['trained_classes'])}")
    if prediction_result['untrained_classes']:
        print(f"   Untrained (estimated): {', '.join(prediction_result['untrained_classes'])}")
    
    # Cuisine recommendation
    print(f"\n{'='*70}")
    print("🍽️  CUISINE RECOMMENDATION")
    print(f"{'='*70}")
    
    if cuisine_info:
        print(f"\n🌏 Region: {cuisine_info['region']}")
        print(f"🗣️  Language: {cuisine_info['language']}")
        print(f"🥘 Cuisine: {cuisine_info['cuisine']}")
        
        print(f"\n📜 MUST-TRY DISHES:")
        for i, dish in enumerate(cuisine_info['dishes'], 1):
            print(f"   {i}. {dish}")
        
        print(f"\n✨ CHARACTERISTICS:")
        for char in cuisine_info['characteristics']:
            print(f"   • {char}")
    else:
        print("\n⚠️  No cuisine information available for this region.")
    
    print(f"\n{'='*70}\n")

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("="*70)
        print("🎯 Robust Native Language Identification")
        print("="*70)
        print("\nUsage: python predict_robust.py <audio_path> [model_path]")
        print("\nExamples:")
        print("  python predict_robust.py test_audio.wav")
        print("  python predict_robust.py test_audio.wav models/hubert_bestlayer_rf_layer8.joblib")
        print("\nFeatures:")
        print("  ✓ Handles untrained classes using similarity matching")
        print("  ✓ Provides confidence scores and warnings")
        print("  ✓ Shows probabilities for all classes")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODELS_DIR / "hubert_bestlayer_rf_layer8.joblib"
    
    # Check if audio exists
    if not Path(audio_path).exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Check device
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    try:
        # Initialize predictor
        predictor = RobustNLIPredictor(model_path)
        
        # Load cuisine mapping
        cuisine_mapping = load_cuisine_mapping()
        
        # Make prediction
        result = predictor.predict(audio_path, device=device, use_all_classes=True)
        
        # Get cuisine info
        cuisine_info = cuisine_mapping.get(result['predicted_class'])
        
        # Display results
        format_output(result, cuisine_info)
        
        # Save results
        output_file = Path(audio_path).stem + "_robust_prediction.json"
        output_data = {
            'audio_file': str(audio_path),
            'prediction': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
            },
            'cuisine_recommendation': cuisine_info
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
