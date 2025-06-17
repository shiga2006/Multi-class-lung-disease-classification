import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# --- Set seeds for reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# --- Config ---
DATA_DIR = "C:/Shiga/Lung Disease Dataset"  # Adjust as needed
IMG_SIZE = 128
N_WAY = 5             # Number of classes per episode (normal, bacterial, viral, covid, tb)
N_SUPPORT = 10        # Number of support images per class
N_QUERY = 15          # Number of query images per class
EPISODES_PER_EPOCH = 500
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names for better visualization
CLASS_NAMES = [
    'Normal',
    'Bacterial Pneumonia',
    'Viral Pneumonia',
    'COVID',
    'Tuberculosis'
]

# --- Simple CNN Encoder ---
class SimpleCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Prototypical Network Definition ---
class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, support, support_labels, query):
        emb_support = self.encoder(support)
        emb_query = self.encoder(query)
        classes = torch.unique(support_labels)
        prototypes = []
        for cls in classes:
            prototypes.append(emb_support[support_labels == cls].mean(0))
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(emb_query, prototypes)
        return -dists  # logits

# --- Episodic Sampler ---
class EpisodicSampler(Dataset):
    def __init__(self, dataset, n_way, n_support, n_query, episodes_per_epoch):
        super().__init__()
        self.dataset = dataset
        self.targets = np.array(dataset.targets)
        self.classes = np.unique(self.targets)
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.class_indices = {int(cls): np.where(self.targets == int(cls))[0] for cls in self.classes}
    def __len__(self):
        return self.episodes_per_epoch
    def __getitem__(self, idx):
        episode_classes = np.random.choice(self.classes, self.n_way, replace=False)
        support_idx, query_idx, support_labels, query_labels = [], [], [], []
        for i, cls in enumerate(episode_classes):
            idxs = np.random.choice(self.class_indices[int(cls)], self.n_support + self.n_query, replace=False)
            support_idx.extend(idxs[:self.n_support])
            query_idx.extend(idxs[self.n_support:])
            support_labels.extend([i] * self.n_support)
            query_labels.extend([i] * self.n_query)
        support_imgs = torch.stack([self.dataset[i][0] for i in support_idx])
        query_imgs = torch.stack([self.dataset[i][0] for i in query_idx])
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        return support_imgs, support_labels, query_imgs, query_labels

# --- Analysis Functions ---
def plot_confusion_matrix(y_true, y_pred, class_names, epoch, phase='val', save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {phase.capitalize()} (Epoch {epoch})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
   
    if save_path:
        plt.savefig(f'{save_path}_confusion_matrix_{phase}_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return cm

def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
   
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
   
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
   
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_report(y_true, y_pred, class_names, epoch, phase='val'):
    """Generate and print detailed classification report"""
    print(f"\n=== Classification Report - {phase.capitalize()} (Epoch {epoch}) ===")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
   
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
   
    # Create a detailed DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
   
    print("\nDetailed Metrics per Class:")
    print(metrics_df.round(4))
   
    # Overall metrics
    macro_avg = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_avg = precision_recall_fscore_support(y_true, y_pred, average='weighted')
   
    print(f"\nMacro Average - Precision: {macro_avg[0]:.4f}, Recall: {macro_avg[1]:.4f}, F1: {macro_avg[2]:.4f}")
    print(f"Weighted Average - Precision: {weighted_avg[0]:.4f}, Recall: {weighted_avg[1]:.4f}, F1: {weighted_avg[2]:.4f}")
   
    return metrics_df

def evaluate_model_detailed(model, data_loader, criterion, device, class_names, epoch, phase='val'):
    """Detailed evaluation with metrics collection"""
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []
    all_labels = []
   
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {phase}"):
            support_imgs, support_labels, query_imgs, query_labels = batch
            support_imgs = support_imgs.squeeze(0).to(device)
            support_labels = support_labels.squeeze(0).to(device)
            query_imgs = query_imgs.squeeze(0).to(device)
            query_labels = query_labels.squeeze(0).to(device)
           
            logits = model(support_imgs, support_labels, query_imgs)
            loss = criterion(logits, query_labels)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == query_labels).float().mean().item()
           
            total_loss += loss.item()
            total_acc += acc
           
            # Store predictions and labels for detailed analysis
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
   
    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
   
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Data loading ---
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_episodic = EpisodicSampler(train_dataset, N_WAY, N_SUPPORT, N_QUERY, EPISODES_PER_EPOCH)
val_episodic = EpisodicSampler(val_dataset, N_WAY, N_SUPPORT, N_QUERY, EPISODES_PER_EPOCH // 10)

train_loader = DataLoader(train_episodic, batch_size=1, shuffle=True)
val_loader = DataLoader(val_episodic, batch_size=1, shuffle=False)

# --- Training ---
if __name__ == "__main__":
    encoder = SimpleCNN(out_dim=64)
    model = ProtoNet(encoder).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
   
    best_val_acc = 0.0
   
    print(f"Training on device: {DEVICE}")
    print(f"Dataset info - Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"Class names: {CLASS_NAMES}")
   
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0
       
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            support_imgs, support_labels, query_imgs, query_labels = batch
            support_imgs = support_imgs.squeeze(0).to(DEVICE)
            support_labels = support_labels.squeeze(0).to(DEVICE)
            query_imgs = query_imgs.squeeze(0).to(DEVICE)
            query_labels = query_labels.squeeze(0).to(DEVICE)
           
            optimizer.zero_grad()
            logits = model(support_imgs, support_labels, query_imgs)
            loss = criterion(logits, query_labels)
            loss.backward()
            optimizer.step()
           
            preds = torch.argmax(logits, dim=1)
            acc = (preds == query_labels).float().mean().item()
            train_loss += loss.item()
            train_acc += acc
           
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
       
        # Detailed validation
        val_loss, val_acc, val_preds, val_labels = evaluate_model_detailed(
            model, val_loader, criterion, DEVICE, CLASS_NAMES, epoch+1, 'validation'
        )
       
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
       
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
              f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
       
        # Generate detailed analysis every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\n{'='*50}")
            print(f"Detailed Analysis - Epoch {epoch+1}")
            print(f"{'='*50}")
           
            # Confusion Matrix
            cm = plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, epoch+1, 'validation', 'protonet_analysis')
           
            # Classification Report
            metrics_df = generate_classification_report(val_labels, val_preds, CLASS_NAMES, epoch+1, 'validation')
           
            print(f"{'='*50}\n")
       
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_protonet_xray_long[5class].pth")
            print(f"Saved best model! New best validation accuracy: {best_val_acc:.4f}")
   
    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL TRAINING ANALYSIS")
    print(f"{'='*60}")
   
    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, 'protonet_analysis')
   
    # Final evaluation on validation set
    print(f"\nFinal validation accuracy: {best_val_acc:.4f}")
   
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_protonet_xray_long[5class].pth"))
    final_val_loss, final_val_acc, final_preds, final_labels = evaluate_model_detailed(
        model, val_loader, criterion, DEVICE, CLASS_NAMES, "Final", 'validation'
    )
   
    # Final confusion matrix and classification report
    plot_confusion_matrix(final_labels, final_preds, CLASS_NAMES, "Final", 'validation', 'protonet_final5')
    generate_classification_report(final_labels, final_preds, CLASS_NAMES, "Final", 'validation')
   
    print(f"\nTraining completed! Best model saved as 'best_protonet_xray_long[5class].pth'")
    print(f"Analysis plots saved with prefix 'protonet_analysis5_' and 'protonet_final5                                                                                                             _'")