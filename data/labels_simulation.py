"""
Simulate human labeling with an oracle.

In real scenarios, humans would label. Here, we use a deterministic oracle
that adds noise to simulate human disagreement/mistakes.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class LabelingOracle:
    """Simulates human labeling with realistic noise patterns."""
    
    def __init__(self, error_rate: float = 0.05, seed: int = 42):
        """
        Initialize oracle.
        
        Args:
            error_rate: Probability of mislabeling (0.05 = 5% error rate)
            seed: Random seed
        """
        self.error_rate = error_rate
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
    
    def label(self, sample_id: str, true_label: int, confidence: float = 1.0) -> int:
        """
        Label a sample with simulated human error.
        
        Args:
            sample_id: Sample identifier
            true_label: Ground truth label
            confidence: Model confidence in the label (0-1)
        
        Returns:
            Noisy label
        """
        # Higher model confidence → lower error rate
        adjusted_error_rate = self.error_rate * (1 - confidence)
        
        if self.rng.random() < adjusted_error_rate:
            # Make a mistake
            return 1 - true_label
        else:
            return true_label
    
    def label_batch(self, samples: List[Dict], confidences: List[float] = None) -> List[int]:
        """
        Label a batch of samples.
        
        Args:
            samples: List of sample dicts with 'label' field
            confidences: Optional model confidences
        
        Returns:
            List of noisy labels
        """
        if confidences is None:
            confidences = [1.0] * len(samples)
        
        labels = []
        for sample, conf in zip(samples, confidences):
            true_label = sample['label']
            noisy_label = self.label(sample['id'], true_label, conf)
            labels.append(noisy_label)
        
        return labels


def load_dataset(metadata_path: str) -> List[Dict]:
    """Load dataset from metadata file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def simulate_labeling(
    dataset_path: str,
    output_path: str,
    error_rate: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    Simulate labeling of entire dataset.
    
    Creates a ground truth file with both true and observed labels.
    """
    dataset = load_dataset(dataset_path)
    oracle = LabelingOracle(error_rate=error_rate, seed=seed)
    
    labels_data = []
    
    print(f"Simulating labeling with {error_rate:.1%} error rate...")
    
    for sample in dataset:
        sample_id = sample['id']
        true_label = sample['label']
        
        # Simulate labeling (assume high confidence for now)
        observed_label = oracle.label(sample_id, true_label, confidence=0.9)
        
        labels_data.append({
            "id": sample_id,
            "true_label": true_label,
            "observed_label": observed_label,
            "agreement": true_label == observed_label,
            "text": sample['text'],
            "image_path": sample['image_path']
        })
    
    # Calculate stats
    num_agreements = sum(1 for l in labels_data if l['agreement'])
    agreement_rate = num_agreements / len(labels_data)
    
    stats = {
        "total_samples": len(labels_data),
        "agreement_rate": agreement_rate,
        "error_rate": 1 - agreement_rate,
        "true_labels_0": sum(1 for l in labels_data if l['true_label'] == 0),
        "true_labels_1": sum(1 for l in labels_data if l['true_label'] == 1),
        "observed_labels_0": sum(1 for l in labels_data if l['observed_label'] == 0),
        "observed_labels_1": sum(1 for l in labels_data if l['observed_label'] == 1),
    }
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "labels": labels_data,
            "stats": stats
        }, f, indent=2)
    
    print(f"✓ Labels simulated: {output_file}")
    print(f"  Agreement rate: {agreement_rate:.1%}")
    print(f"  Total samples: {len(labels_data)}")
    
    return stats


def get_label(sample_id: str, labels_file: str) -> Tuple[int, int, bool]:
    """
    Get label for a sample.
    
    Returns:
        (true_label, observed_label, agreement)
    """
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    for label_entry in data['labels']:
        if label_entry['id'] == sample_id:
            return (
                label_entry['true_label'],
                label_entry['observed_label'],
                label_entry['agreement']
            )
    
    raise ValueError(f"Sample {sample_id} not found in labels file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate labeling")
    parser.add_argument("--dataset_path", type=str, default="data/dataset.json", help="Dataset file")
    parser.add_argument("--output_path", type=str, default="data/labels_oracle.json", help="Output file")
    parser.add_argument("--error_rate", type=float, default=0.05, help="Simulated error rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    stats = simulate_labeling(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        error_rate=args.error_rate,
        seed=args.seed
    )
    
    print("\nLabeling Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
