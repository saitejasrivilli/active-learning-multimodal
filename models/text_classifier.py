"""
Text classifier for toxicity detection.

Fine-tuned BERT on content moderation task.
Outputs: logits, probabilities, confidence scores.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import pickle


class TextDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get('text', '')
        label = sample.get('label', 0)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'id': sample.get('id', '')
        }


class TextClassifier(nn.Module):
    """BERT-based text classifier."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        logits = self.classifier(self.dropout(pooled_output))
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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
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
    output_dir: str = "models/text",
    device: str = 'cuda',
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    model_name: str = 'distilbert-base-uncased'
):
    """Train text classifier."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    train_samples = load_dataset(dataset_path, 'train', split_file)
    val_samples = load_dataset(dataset_path, 'val', split_file)
    
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Datasets
    train_dataset = TextDataset(train_samples, tokenizer)
    val_dataset = TextDataset(val_samples, tokenizer)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = TextClassifier(model_name=model_name, num_classes=2)
    model.to(device)
    
    # Optimizer
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nTraining text classifier ({model_name})...")
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
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path / 'tokenizer')
    
    # Save config
    config = {
        'model_name': model_name,
        'num_classes': 2,
        'max_length': 128,
        'device': device,
        'history': history,
        'best_val_acc': float(best_val_acc)
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Model saved to {output_path}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    return model, tokenizer


class TextClassifierInference:
    """Inference wrapper for text classifier."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        
        # Load config
        with open(Path(model_path) / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_path) / 'tokenizer')
        
        # Load model
        self.model = TextClassifier(
            model_name=self.config['model_name'],
            num_classes=self.config['num_classes']
        )
        self.model.load_state_dict(torch.load(Path(model_path) / 'best_model.pt', map_location=device))
        self.model.to(device)
        self.model.eval()
    
    def predict_batch(self, texts: List[str]) -> Dict:
        """
        Predict for batch of texts.
        
        Returns:
            {
                'logits': tensor,
                'probs': tensor,
                'predictions': list,
                'confidences': list,
                'entropies': list
            }
        """
        with torch.no_grad():
            encodings = self.tokenizer(
                texts,
                max_length=self.config['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            
            # Predictions and confidence
            preds = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]
            
            # Entropy (uncertainty)
            entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            return {
                'logits': logits.cpu().numpy(),
                'probs': probs.cpu().numpy(),
                'predictions': preds.cpu().numpy(),
                'confidences': confidences.cpu().numpy(),
                'entropies': entropies.cpu().numpy()
            }
    
    def predict_single(self, text: str) -> Dict:
        """Predict for single text."""
        result = self.predict_batch([text])
        return {k: v[0] for k, v in result.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train text classifier")
    parser.add_argument("--dataset_path", type=str, default="data/dataset.json", help="Dataset file")
    parser.add_argument("--split_file", type=str, default="data/splits.json", help="Splits file")
    parser.add_argument("--output_dir", type=str, default="models/text", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    model, tokenizer = train_classifier(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model_name
    )
