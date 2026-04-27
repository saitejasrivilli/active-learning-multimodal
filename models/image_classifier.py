"""
Image classifier for content safety.

Uses CLIP with fine-tuning for safety classification.
Outputs: logits, probabilities, confidence scores.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import clip


class ImageDataset(Dataset):
    """Dataset for image classification."""
    
    def __init__(self, samples: List[Dict], image_preprocessor, transform=None):
        self.samples = samples
        self.image_preprocessor = image_preprocessor
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Fallback: create blank image
            image = Image.new('RGB', (224, 224), color='white')
        
        # Preprocess
        image = self.image_preprocessor(image)
        label = sample.get('label', 0)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'id': sample.get('id', ''),
            'image_path': image_path
        }


class ImageClassifier(nn.Module):
    """CLIP-based image classifier."""
    
    def __init__(self, clip_model, num_classes: int = 2, freeze_backbone: bool = False):
        super().__init__()
        self.clip_model = clip_model
        self.frozen = freeze_backbone
        
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Linear head on top of CLIP embeddings
        self.vision_dim = clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images):
        # Get image embeddings from CLIP
        image_features = self.clip_model.encode_image(images)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Classify
        logits = self.classifier(image_features)
        return logits


def load_dataset(dataset_path: str, split: str = 'train', split_file: str = None) -> List[Dict]:
    """Load dataset samples for a specific split."""
    with open(dataset_path, 'r') as f:
        all_samples = json.load(f)
    
    if split_file:
        with open(split_file, 'r') as f:
            splits = json.load(f)
        indices = splits[split]['indices']
        return [all_samples[i] for i in indices]
    
    # Default split
    n = len(all_samples)
    if split == 'train':
        return all_samples[:int(0.7 * n)]
    elif split == 'val':
        return all_samples[int(0.7 * n):int(0.85 * n)]
    else:  # test
        return all_samples[int(0.85 * n):]


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def train_classifier(
    dataset_path: str,
    split_file: str = None,
    output_dir: str = "models/image",
    device: str = 'cuda',
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    freeze_backbone: bool = False,
    clip_model_name: str = "ViT-B/32"
):
    """Train image classifier."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load CLIP model
    print(f"Loading CLIP model ({clip_model_name})...")
    clip_model, image_preprocessor = clip.load(clip_model_name, device=device)
    
    # Load data
    print("Loading dataset...")
    train_samples = load_dataset(dataset_path, 'train', split_file)
    val_samples = load_dataset(dataset_path, 'val', split_file)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    
    # Datasets
    train_dataset = ImageDataset(train_samples, image_preprocessor)
    val_dataset = ImageDataset(val_samples, image_preprocessor)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = ImageClassifier(clip_model, num_classes=2, freeze_backbone=freeze_backbone)
    model.to(device)
    
    # Optimizer
    if freeze_backbone:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining image classifier...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / 'best_model.pt')
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), output_path / 'final_model.pt')
    
    # Save config
    config = {
        'clip_model_name': clip_model_name,
        'num_classes': 2,
        'freeze_backbone': freeze_backbone,
        'device': device,
        'history': history,
        'best_val_acc': float(best_val_acc)
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Model saved to {output_path}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    return model, image_preprocessor


class ImageClassifierInference:
    """Inference wrapper for image classifier."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        
        # Load config
        with open(Path(model_path) / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load CLIP
        self.clip_model, self.image_preprocessor = clip.load(
            self.config['clip_model_name'],
            device=device
        )
        
        # Load model
        self.model = ImageClassifier(
            self.clip_model,
            num_classes=2,
            freeze_backbone=self.config['freeze_backbone']
        )
        self.model.load_state_dict(torch.load(Path(model_path) / 'best_model.pt', map_location=device))
        self.model.to(device)
        self.model.eval()
    
    def predict_batch(self, images: List[Image.Image]) -> Dict:
        """
        Predict for batch of images.
        
        Returns:
            {
                'logits': array,
                'probs': array,
                'predictions': array,
                'confidences': array,
                'entropies': array
            }
        """
        with torch.no_grad():
            # Preprocess
            processed = torch.stack([
                self.image_preprocessor(img.convert('RGB')) if isinstance(img, Image.Image)
                else self.image_preprocessor(img)
                for img in images
            ]).to(self.device)
            
            logits = self.model(processed)
            probs = torch.softmax(logits, dim=1)
            
            # Predictions and confidence
            preds = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            
            # Entropy
            entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            return {
                'logits': logits.cpu().numpy(),
                'probs': probs.cpu().numpy(),
                'predictions': preds.cpu().numpy(),
                'confidences': confidences.cpu().numpy(),
                'entropies': entropies.cpu().numpy()
            }
    
    def predict_single(self, image: Image.Image) -> Dict:
        """Predict for single image."""
        result = self.predict_batch([image])
        return {k: v[0] for k, v in result.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--dataset_path", type=str, default="data/dataset.json", help="Dataset file")
    parser.add_argument("--split_file", type=str, default="data/splits.json", help="Splits file")
    parser.add_argument("--output_dir", type=str, default="models/image", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze CLIP backbone")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    model, preprocessor = train_classifier(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_backbone=args.freeze_backbone,
        clip_model_name=args.clip_model
    )
