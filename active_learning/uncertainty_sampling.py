"""
Uncertainty sampling strategies for active learning.

Selects samples that the model is most uncertain about.
Key idea: high uncertainty = high information value
"""

import numpy as np
import torch
from typing import List, Tuple
from scipy.stats import entropy as scipy_entropy


class UncertaintySampler:
    """Base uncertainty sampler."""
    
    def __init__(self, method: str = 'entropy'):
        """
        Initialize sampler.
        
        Args:
            method: 'entropy', 'margin', 'bald'
        """
        self.method = method
    
    def compute_uncertainty(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores.
        
        Args:
            probs: [num_samples, num_classes] probability predictions
        
        Returns:
            [num_samples] uncertainty scores (higher = more uncertain)
        """
        if self.method == 'entropy':
            return self._entropy(probs)
        elif self.method == 'margin':
            return self._margin(probs)
        elif self.method == 'bald':
            return self._bald(probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Shannon entropy.
        
        H(p) = -sum(p * log(p))
        Maximum when uniform distribution (p=[0.5, 0.5] for binary)
        """
        # Avoid log(0)
        probs_safe = np.clip(probs, 1e-10, 1 - 1e-10)
        entropy = -np.sum(probs * np.log(probs_safe), axis=1)
        
        # Normalize to [0, 1] for binary classification
        # Max entropy for binary is log(2) ≈ 0.693
        max_entropy = np.log(probs.shape[1])
        entropy_normalized = entropy / max_entropy
        
        return entropy_normalized
    
    def _margin(self, probs: np.ndarray) -> np.ndarray:
        """
        Margin sampling.
        
        Uncertainty = 1 - (max_prob - second_max_prob)
        When max_prob ≈ second_max_prob, uncertainty is high.
        """
        # Get top 2 probabilities
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        max_prob = sorted_probs[:, 0]
        second_max = sorted_probs[:, 1]
        
        margin = max_prob - second_max
        uncertainty = 1 - margin
        
        return uncertainty
    
    def _bald(self, probs: np.ndarray) -> np.ndarray:
        """
        BALD (Bayesian Active Learning by Disagreement).
        
        For single model (not ensemble), approximates as entropy.
        Full BALD requires multiple forward passes (MC dropout).
        """
        # Approximation: use entropy
        return self._entropy(probs)
    
    def select_batch(
        self,
        probs: np.ndarray,
        num_samples: int,
        exclude_indices: List[int] = None
    ) -> List[int]:
        """
        Select most uncertain samples.
        
        Args:
            probs: Probability predictions
            num_samples: How many to select
            exclude_indices: Indices to exclude (already labeled)
        
        Returns:
            Indices of selected samples
        """
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(probs)
        
        # Exclude already labeled
        if exclude_indices:
            uncertainty[exclude_indices] = -np.inf
        
        # Select top-k
        selected = np.argsort(uncertainty)[::-1][:num_samples]
        
        return selected.tolist()


class EntropySampler(UncertaintySampler):
    """Entropy-based uncertainty sampling."""
    
    def __init__(self):
        super().__init__(method='entropy')


class MarginSampler(UncertaintySampler):
    """Margin-based uncertainty sampling."""
    
    def __init__(self):
        super().__init__(method='margin')


class MultimodalUncertaintySampler:
    """
    Combines uncertainty from text and image modalities.
    
    Three strategies:
    1. Max uncertainty: max(text_unc, image_unc)
    2. Average uncertainty: (text_unc + image_unc) / 2
    3. Product uncertainty: text_unc * image_unc (more selective)
    """
    
    def __init__(self, strategy: str = 'max'):
        """
        Args:
            strategy: 'max', 'average', 'product'
        """
        self.strategy = strategy
        self.text_sampler = EntropySampler()
        self.image_sampler = EntropySampler()
    
    def compute_joint_uncertainty(
        self,
        text_probs: np.ndarray,
        image_probs: np.ndarray
    ) -> np.ndarray:
        """
        Combine text and image uncertainties.
        
        Args:
            text_probs: [num_samples, 2]
            image_probs: [num_samples, 2]
        
        Returns:
            [num_samples] joint uncertainty
        """
        text_unc = self.text_sampler.compute_uncertainty(text_probs)
        image_unc = self.image_sampler.compute_uncertainty(image_probs)
        
        if self.strategy == 'max':
            joint_unc = np.maximum(text_unc, image_unc)
        elif self.strategy == 'average':
            joint_unc = (text_unc + image_unc) / 2
        elif self.strategy == 'product':
            # Emphasize samples uncertain in both modalities
            joint_unc = text_unc * image_unc
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return joint_unc
    
    def select_batch(
        self,
        text_probs: np.ndarray,
        image_probs: np.ndarray,
        num_samples: int,
        exclude_indices: List[int] = None
    ) -> List[int]:
        """Select most uncertain samples across modalities."""
        joint_unc = self.compute_joint_uncertainty(text_probs, image_probs)
        
        if exclude_indices:
            joint_unc[exclude_indices] = -np.inf
        
        selected = np.argsort(joint_unc)[::-1][:num_samples]
        return selected.tolist()


def analyze_uncertainty_distribution(
    probs: np.ndarray,
    method: str = 'entropy',
    bins: int = 20
) -> dict:
    """
    Analyze uncertainty distribution.
    
    Useful for understanding model confidence patterns.
    """
    sampler = UncertaintySampler(method=method)
    uncertainty = sampler.compute_uncertainty(probs)
    
    stats = {
        'mean': float(np.mean(uncertainty)),
        'std': float(np.std(uncertainty)),
        'min': float(np.min(uncertainty)),
        'max': float(np.max(uncertainty)),
        'median': float(np.median(uncertainty)),
        'q25': float(np.percentile(uncertainty, 25)),
        'q75': float(np.percentile(uncertainty, 75)),
        'percentiles': {
            'p10': float(np.percentile(uncertainty, 10)),
            'p50': float(np.percentile(uncertainty, 50)),
            'p90': float(np.percentile(uncertainty, 90)),
        }
    }
    
    # Count samples in different confidence ranges
    high_conf = np.sum(uncertainty < 0.3)  # Confident
    medium_conf = np.sum((uncertainty >= 0.3) & (uncertainty < 0.7))  # Medium
    low_conf = np.sum(uncertainty >= 0.7)  # Uncertain
    
    stats['high_confidence'] = int(high_conf)
    stats['medium_confidence'] = int(medium_conf)
    stats['low_confidence'] = int(low_conf)
    
    return stats


def compute_confidence_bins(probs: np.ndarray, num_bins: int = 10) -> dict:
    """
    Bin samples by confidence level.
    
    Useful for seeing confidence distribution.
    """
    confidences = np.max(probs, axis=1)
    
    bins_dict = {}
    for i in range(num_bins):
        lower = i / num_bins
        upper = (i + 1) / num_bins
        
        mask = (confidences >= lower) & (confidences < upper)
        count = np.sum(mask)
        
        bins_dict[f'{lower:.1f}-{upper:.1f}'] = int(count)
    
    return bins_dict


if __name__ == "__main__":
    # Example
    print("Uncertainty Sampling Module\n")
    
    # Simulate predictions
    num_samples = 1000
    probs = np.random.dirichlet([1, 1], size=num_samples)
    
    # Test different methods
    for method in ['entropy', 'margin', 'bald']:
        sampler = UncertaintySampler(method=method)
        uncertainty = sampler.compute_uncertainty(probs)
        
        print(f"{method.upper()}:")
        print(f"  Mean: {np.mean(uncertainty):.4f}")
        print(f"  Std: {np.std(uncertainty):.4f}")
        print(f"  Selected (top 100):")
        selected = sampler.select_batch(probs, num_samples=100)
        print(f"    Indices: {selected[:10]}...")
        print()
    
    # Multimodal
    print("Multimodal Uncertainty:")
    text_probs = np.random.dirichlet([1, 1], size=num_samples)
    image_probs = np.random.dirichlet([1, 1], size=num_samples)
    
    for strategy in ['max', 'average', 'product']:
        sampler = MultimodalUncertaintySampler(strategy=strategy)
        selected = sampler.select_batch(text_probs, image_probs, num_samples=100)
        print(f"  {strategy}: {len(selected)} selected")
