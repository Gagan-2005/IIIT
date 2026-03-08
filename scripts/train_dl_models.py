"""
Deep Learning Models for Native Language Identification
Implements CNN, BiLSTM, and Transformer architectures for comparison with RF baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_LAYER = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

print(f"Using device: {DEVICE}")

# ============================================================================
# Dataset Class
# ============================================================================

class NLIDataset(Dataset):
    """Dataset for Native Language Identification"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# Model Architectures
# ============================================================================

class CNN_NLI(nn.Module):
    """
    1D CNN for Native Language Identification
    Treats feature vector as 1D sequence
    """
    
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(CNN_NLI, self).__init__()
        
        # Reshape input to (batch, channels=1, sequence_length)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        self.flat_size = 256 * (input_dim // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Conv blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class BiLSTM_NLI(nn.Module):
    """
    Bidirectional LSTM for Native Language Identification
    Processes feature vector as sequence
    """
    
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, dropout=0.5):
        super(BiLSTM_NLI, self).__init__()
        
        # Treat input as sequence: (batch, seq_len=input_dim, features=1)
        # Or chunk into segments: (batch, seq_len, features)
        self.chunk_size = 16  # Chunk 768-dim into 16 features of 48-dim each
        self.num_chunks = input_dim // self.chunk_size
        
        self.lstm = nn.LSTM(
            input_size=self.chunk_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, input_dim) -> (batch, num_chunks, chunk_size)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_chunks, self.chunk_size)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from both directions
        h_n = h_n.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
        forward_hidden = h_n[-1, 0, :, :]
        backward_hidden = h_n[-1, 1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Fully connected
        x = F.relu(self.fc1(hidden))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class TransformerEncoder_NLI(nn.Module):
    """
    Transformer Encoder for Native Language Identification
    Uses self-attention over feature sequence
    """
    
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4, dropout=0.3):
        super(TransformerEncoder_NLI, self).__init__()
        
        # Chunk input into patches
        self.chunk_size = 32
        self.num_chunks = input_dim // self.chunk_size
        
        # Embedding layer
        self.embedding = nn.Linear(self.chunk_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_chunks, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(d_model, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (batch, input_dim) -> (batch, num_chunks, chunk_size)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_chunks, self.chunk_size)
        
        # Embed and add positional encoding
        x = self.embedding(x) + self.pos_encoding
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (total_loss / len(dataloader), 
            100. * correct / total,
            np.array(all_preds),
            np.array(all_labels))

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, model_name):
    """Complete training loop"""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print(f"\nTraining {model_name}...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Record
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
    
    print(f"\n{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return train_losses, train_accs, val_losses, val_accs, best_val_acc

# ============================================================================
# Main Execution
# ============================================================================

def load_hubert_features(layer=BEST_LAYER, feature_dir='features/hubert'):
    """Load HuBERT features from .npz files"""
    print(f"\n[1/5] Loading HuBERT features from Layer {layer}...")
    
    features = []
    labels = []
    file_ids = []
    
    feature_path = Path(feature_dir)
    npz_files = list(feature_path.glob('*.npz'))
    
    print(f"  Found {len(npz_files)} .npz files")
    
    # Label mapping from filename patterns
    label_keywords = {
        'andhra': 'andhra_pradesh',
        'gujrat': 'gujrat',
        'jharkhand': 'jharkhand', 
        'karnataka': 'karnataka',
        'kerala': 'kerala',
        'tamil': 'tamil'
    }
    
    skipped = 0
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract pooled features for specified layer
            if 'pooled' in data:
                pooled = data['pooled']
                features.append(pooled[layer])
                
                # Extract label from filename
                filename_lower = npz_file.stem.lower()
                label = None
                for keyword, lang in label_keywords.items():
                    if keyword in filename_lower:
                        label = lang
                        break
                
                if label is None:
                    skipped += 1
                    features.pop()  # Remove the feature we just added
                    continue
                    
                labels.append(label)
                file_ids.append(npz_file.stem)
        except Exception as e:
            skipped += 1
            continue
    
    X = np.array(features)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"  ✓ Loaded {len(X)} samples (skipped {skipped})")
    print(f"  Feature shape: {X.shape}")
    print(f"  Classes: {list(le.classes_)}")
    print("  Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(le.inverse_transform(unique), counts):
        print(f"    {cls}: {count}")
    
    return X, y, le, file_ids

def main():
    print("="*70)
    print("Deep Learning Models for NLI - CNN, BiLSTM, Transformer")
    print("="*70)
    
    # Load actual HuBERT features
    X, y, label_encoder, file_ids = load_hubert_features()
    
    # Split data
    print(f"\n[2/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Save scaler and label encoder for inference
    joblib.dump(scaler, 'models/dl_scaler.joblib')
    joblib.dump(label_encoder, 'models/dl_label_encoder.joblib')
    print(f"  ✓ Saved scaler and label encoder")
    
    # Create datasets
    train_dataset = NLIDataset(X_train, y_train)
    val_dataset = NLIDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"  Input dim: {input_dim}, Classes: {num_classes}")
    
    # Define models
    models = {
        'CNN': CNN_NLI(input_dim, num_classes),
        'BiLSTM': BiLSTM_NLI(input_dim, num_classes),
        'Transformer': TransformerEncoder_NLI(input_dim, num_classes)
    }
    
    results = {}
    
    # Train each model
    for idx, (model_name, model) in enumerate(models.items(), start=3):
        print(f"\n{'='*70}")
        print(f"[{idx}/5] Training {model_name}")
        print(f"{'='*70}")
        
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train
        train_losses, train_accs, val_losses, val_accs, best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            EPOCHS, DEVICE, model_name
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f'models/{model_name}_best.pth'))
        model.eval()
        
        # Final evaluation
        _, final_acc, y_pred, y_true = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"\n  Final Validation Results:")
        print(f"    Accuracy: {final_acc:.2f}%")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=label_encoder.classes_,
                                   zero_division=0))
        
        # Store results
        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'final_acc': final_acc
        }
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"\n{'Model':<15} {'Best Val Accuracy':<20}")
    print("-" * 35)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['best_val_acc']:>17.2f}%")
    
    print(f"\n{'='*70}\n")
    
    # Save results to JSON
    import json
    results_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'device': str(DEVICE),
        'best_layer': BEST_LAYER,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_samples': len(X),
        'num_classes': num_classes,
        'classes': label_encoder.classes_.tolist(),
        'models': {}
    }
    
    for model_name, result in results.items():
        results_summary['models'][model_name] = {
            'best_val_acc': float(result['best_val_acc']),
            'final_acc': float(result['final_acc'])
        }
    
    with open('results/dl_models_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("✓ Saved: results/dl_models_results.json")
    
    # Plot comparison
    print("\n[5/5] Generating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy curves
    for model_name, result in results.items():
        axes[0, 0].plot(result['val_accs'], label=model_name, linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Model Comparison - Validation Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    for model_name, result in results.items():
        axes[0, 1].plot(result['val_losses'], label=model_name, linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Model Comparison - Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bar chart comparison
    model_names = list(results.keys())
    best_accs = [results[m]['best_val_acc'] for m in model_names]
    axes[1, 0].bar(model_names, best_accs, color=['#2ecc71', '#3498db', '#e74c3c'])
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Best Accuracy Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(best_accs):
        axes[1, 0].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # Training vs Validation curves (overfitting check)
    for model_name, result in results.items():
        train_accs = result['train_accs']
        val_accs = result['val_accs']
        gap = [t - v for t, v in zip(train_accs, val_accs)]
        axes[1, 1].plot(gap, label=model_name, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train - Val Accuracy Gap (%)')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dl_models_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/dl_models_comparison.png")
    
    print("\n" + "="*70)
    print("✅ Deep learning model training complete!")
    print("="*70)
    print(f"\nBest Model: {max(results.items(), key=lambda x: x[1]['best_val_acc'])[0]}")
    print(f"Best Accuracy: {max(r['best_val_acc'] for r in results.values()):.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
