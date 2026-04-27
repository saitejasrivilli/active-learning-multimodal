"""
Multimodal fusion for content moderation.

Combines text and image predictions using attention-based fusion.
Outperforms single-modality approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
from pathlib import Path
import json


class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion."""
    
    def __init__(self, text_dim: int = 2, image_dim: int = 2, hidden_dim: int = 128):
        """
        Initialize attention fusion.
        
        Args:
            text_dim: Dimension of text logits/embeddings
            image_dim: Dimension of image logits/embeddings
            hidden_dim: Hidden dimension for fusion
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Text branch attention
        self.text_attention = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive weights
        )
        
        # Image branch attention
        self.image_attention = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Output logits for 2 classes
        )
    
    def forward(self, text_logits, image_logits):
        """
        Fuse text and image predictions.
        
        Args:
            text_logits: [batch_size, 2]
            image_logits: [batch_size, 2]
        
        Returns:
            fused_logits: [batch_size, 2]
        """
        # Compute attention weights
        text_weight = self.text_attention(text_logits)  # [batch_size, 1]
        image_weight = self.image_attention(image_logits)  # [batch_size, 1]
        
        # Normalize weights
        total_weight = text_weight + image_weight + 1e-8
        text_weight = text_weight / total_weight
        image_weight = image_weight / total_weight
        
        # Weighted combination
        weighted_text = text_logits * text_weight
        weighted_image = image_logits * image_weight
        
        # Concatenate for fusion
        fused = torch.cat([weighted_text, weighted_image], dim=1)
        
        # Final fusion layer
        fused_logits = self.fusion(fused)
        
        return fused_logits


class SimpleWeightedFusion(nn.Module):
    """Simple weighted fusion (learnable weights)."""
    
    def __init__(self):
        super().__init__()
        self.text_weight = nn.Parameter(torch.tensor(0.5))
        self.image_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, text_logits, image_logits):
        """Simple weighted average."""
        text_weight = torch.sigmoid(self.text_weight)
        image_weight = torch.sigmoid(self.image_weight)
        
        # Normalize
        total = text_weight + image_weight
        text_weight = text_weight / total
        image_weight = image_weight / total
        
        # Weighted sum
        fused = text_weight * text_logits + image_weight * image_logits
        return fused


class MultimodalClassifier:
    """High-level multimodal classification interface."""
    
    def __init__(
        self,
        text_classifier,
        image_classifier,
        fusion_type: str = 'attention',
        device: str = 'cuda'
    ):
        """
        Initialize multimodal classifier.
        
        Args:
            text_classifier: Text classifier inference object
            image_classifier: Image classifier inference object
            fusion_type: 'attention' or 'weighted'
            device: Device to use
        """
        self.text_classifier = text_classifier
        self.image_classifier = image_classifier
        self.device = device
        
        # Initialize fusion
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(text_dim=2, image_dim=2)
        elif fusion_type == 'weighted':
            self.fusion = SimpleWeightedFusion()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion.to(device)
        self.fusion_type = fusion_type
    
    def predict(self, texts: List[str], images: List) -> Dict:
        """
        Multimodal prediction.
        
        Args:
            texts: List of texts
            images: List of PIL images
        
        Returns:
            {
                'text_probs': array,
                'image_probs': array,
                'fused_probs': array,
                'fused_predictions': array,
                'fused_confidences': array,
                'fusion_weights': dict
            }
        """
        # Text predictions
        text_result = self.text_classifier.predict_batch(texts)
        text_logits = torch.from_numpy(text_result['logits']).float().to(self.device)
        text_probs = torch.softmax(text_logits, dim=1)
        
        # Image predictions
        image_result = self.image_classifier.predict_batch(images)
        image_logits = torch.from_numpy(image_result['logits']).float().to(self.device)
        image_probs = torch.softmax(image_logits, dim=1)
        
        # Fusion
        self.fusion.eval()
        with torch.no_grad():
            fused_logits = self.fusion(text_logits, image_logits)
            fused_probs = torch.softmax(fused_logits, dim=1)
            
            fused_preds = torch.argmax(fused_probs, dim=1)
            fused_confs = torch.max(fused_probs, dim=1)[0]
        
        # Get fusion weights (for attention)
        fusion_weights = {}
        if self.fusion_type == 'attention':
            with torch.no_grad():
                text_weight = self.fusion.text_attention(text_logits)
                image_weight = self.fusion.image_attention(image_logits)
                total_weight = text_weight + image_weight + 1e-8
                fusion_weights['text'] = (text_weight / total_weight).cpu().mean().item()
                fusion_weights['image'] = (image_weight / total_weight).cpu().mean().item()
        
        return {
            'text_probs': text_probs.cpu().numpy(),
            'text_predictions': text_result['predictions'],
            'text_confidences': text_result['confidences'],
            'image_probs': image_probs.cpu().numpy(),
            'image_predictions': image_result['predictions'],
            'image_confidences': image_result['confidences'],
            'fused_probs': fused_probs.cpu().numpy(),
            'fused_predictions': fused_preds.cpu().numpy(),
            'fused_confidences': fused_confs.cpu().numpy(),
            'fusion_weights': fusion_weights
        }
    
    def get_uncertainty(self, texts: List[str], images: List) -> np.ndarray:
        """
        Get uncertainty scores (entropy) from multimodal prediction.
        
        Higher entropy = more uncertain.
        """
        result = self.predict(texts, images)
        probs = result['fused_probs']
        
        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy
    
    def train_fusion(self, text_logits, image_logits, labels, epochs: int = 10, lr: float = 1e-3):
        """
        Train fusion layer on labeled data.
        
        Args:
            text_logits: [num_samples, 2] text predictions
            image_logits: [num_samples, 2] image predictions
            labels: [num_samples] ground truth labels
            epochs: Number of training epochs
            lr: Learning rate
        """
        text_logits = torch.from_numpy(text_logits).float().to(self.device)
        image_logits = torch.from_numpy(image_logits).float().to(self.device)
        labels = torch.from_numpy(labels).long().to(self.device)
        
        optimizer = torch.optim.Adam(self.fusion.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.fusion.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            fused_logits = self.fusion(text_logits, image_logits)
            loss = criterion(fused_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                preds = torch.argmax(fused_logits, dim=1)
                acc = (preds == labels).float().mean().item()
                print(f"Epoch {epoch + 1}: loss={loss.item():.4f}, acc={acc:.4f}")
        
        self.fusion.eval()
        print("✓ Fusion training complete")


def compare_fusion_strategies(text_logits, image_logits, labels):
    """
    Compare different fusion strategies.
    
    Returns:
        Comparison results
    """
    results = {}
    
    # Convert to tensors
    text_logits_t = torch.from_numpy(text_logits).float()
    image_logits_t = torch.from_numpy(image_logits).float()
    labels_t = torch.from_numpy(labels).long()
    
    # 1. Text only
    text_preds = np.argmax(text_logits, axis=1)
    text_acc = np.mean(text_preds == labels)
    results['text_only'] = {'accuracy': float(text_acc)}
    
    # 2. Image only
    image_preds = np.argmax(image_logits, axis=1)
    image_acc = np.mean(image_preds == labels)
    results['image_only'] = {'accuracy': float(image_acc)}
    
    # 3. Simple average
    avg_logits = (text_logits + image_logits) / 2
    avg_preds = np.argmax(avg_logits, axis=1)
    avg_acc = np.mean(avg_preds == labels)
    results['average'] = {'accuracy': float(avg_acc)}
    
    # 4. Weighted (random weights for comparison)
    weighted_logits = 0.6 * text_logits + 0.4 * image_logits
    weighted_preds = np.argmax(weighted_logits, axis=1)
    weighted_acc = np.mean(weighted_preds == labels)
    results['weighted'] = {'accuracy': float(weighted_acc), 'text_weight': 0.6, 'image_weight': 0.4}
    
    # 5. Max confidence
    text_conf = np.max(torch.softmax(text_logits_t, dim=1).numpy(), axis=1)
    image_conf = np.max(torch.softmax(image_logits_t, dim=1).numpy(), axis=1)
    max_conf_preds = text_preds.copy()
    mask = image_conf > text_conf
    max_conf_preds[mask] = image_preds[mask]
    max_conf_acc = np.mean(max_conf_preds == labels)
    results['max_confidence'] = {'accuracy': float(max_conf_acc)}
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Multimodal Fusion Module")
    
    # Simulate predictions
    batch_size = 32
    text_logits = np.random.randn(batch_size, 2)
    image_logits = np.random.randn(batch_size, 2)
    labels = np.random.randint(0, 2, batch_size)
    
    # Compare strategies
    results = compare_fusion_strategies(text_logits, image_logits, labels)
    
    print("\nFusion Strategy Comparison:")
    for strategy, result in results.items():
        print(f"  {strategy}: {result}")
