"""
Ranking strategy combining uncertainty and diversity for sample selection.

This is the core of the active learning system: rank all unlabeled samples
by their information value (uncertainty × diversity) and label the top-k.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


@dataclass
class RankedSample:
    """A sample with its ranking scores."""
    index: int
    sample_id: str
    uncertainty: float
    diversity: float
    combined_score: float
    rank: int
    
    def to_dict(self):
        return {
            'index': self.index,
            'sample_id': self.sample_id,
            'uncertainty': float(self.uncertainty),
            'diversity': float(self.diversity),
            'combined_score': float(self.combined_score),
            'rank': self.rank
        }


class RankingStrategy:
    """Core ranking strategy for active learning."""
    
    def __init__(
        self,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
        diversity_method: str = 'distance'
    ):
        """
        Initialize ranking strategy.
        
        Args:
            uncertainty_weight: Weight for uncertainty (0.0-1.0)
            diversity_weight: Weight for diversity (0.0-1.0)
            diversity_method: 'distance', 'clustering', or 'hybrid'
        """
        assert uncertainty_weight + diversity_weight > 0, "Weights must sum > 0"
        
        # Normalize weights
        total = uncertainty_weight + diversity_weight
        self.uncertainty_weight = uncertainty_weight / total
        self.diversity_weight = diversity_weight / total
        self.diversity_method = diversity_method
    
    def rank_samples(
        self,
        sample_ids: List[str],
        uncertainty_scores: np.ndarray,
        embeddings: np.ndarray,
        labeled_indices: List[int] = None,
        diversity_scores: np.ndarray = None
    ) -> List[RankedSample]:
        """
        Rank all samples by information value.
        
        Args:
            sample_ids: Sample identifiers
            uncertainty_scores: [num_samples] uncertainty from model
            embeddings: [num_samples, embedding_dim] sample embeddings
            labeled_indices: Indices already labeled (excluded)
            diversity_scores: Optional pre-computed diversity scores
        
        Returns:
            List of RankedSample objects, sorted by combined_score (descending)
        """
        num_samples = len(sample_ids)
        
        # Normalize uncertainty
        uncertainty_norm = self._normalize(uncertainty_scores)
        
        # Compute diversity if not provided
        if diversity_scores is None:
            diversity_scores = self._compute_diversity_scores(embeddings)
        
        diversity_norm = self._normalize(diversity_scores)
        
        # Combine scores
        combined_scores = (
            self.uncertainty_weight * uncertainty_norm +
            self.diversity_weight * diversity_norm
        )
        
        # Mark labeled as invalid
        if labeled_indices:
            combined_scores[labeled_indices] = -np.inf
        
        # Rank
        ranked_samples = []
        for rank, idx in enumerate(np.argsort(combined_scores)[::-1]):
            if np.isinf(combined_scores[idx]):
                break
            
            ranked_sample = RankedSample(
                index=idx,
                sample_id=sample_ids[idx],
                uncertainty=float(uncertainty_norm[idx]),
                diversity=float(diversity_norm[idx]),
                combined_score=float(combined_scores[idx]),
                rank=rank + 1
            )
            ranked_samples.append(ranked_sample)
        
        return ranked_samples
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val == min_val:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_val) / (max_val - min_val)
    
    def _compute_diversity_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute diversity as distance to nearest neighbor.
        
        High diversity = far from others
        """
        num_samples = embeddings.shape[0]
        k = min(5, num_samples // 10)
        
        diversity_scores = np.zeros(num_samples)
        
        for i in range(num_samples):
            distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
            distances[i] = np.inf  # Exclude self
            
            if np.sum(np.isfinite(distances)) >= k:
                nearest_k = np.sort(distances)[:k]
                diversity_scores[i] = np.mean(nearest_k)
        
        return diversity_scores
    
    def select_batch(
        self,
        sample_ids: List[str],
        uncertainty_scores: np.ndarray,
        embeddings: np.ndarray,
        num_samples: int,
        labeled_indices: List[int] = None
    ) -> Tuple[List[int], List[RankedSample]]:
        """
        Select top-k samples for labeling.
        
        Returns:
            (selected_indices, ranked_samples_top_k)
        """
        ranked = self.rank_samples(
            sample_ids=sample_ids,
            uncertainty_scores=uncertainty_scores,
            embeddings=embeddings,
            labeled_indices=labeled_indices
        )
        
        # Take top-k
        selected_ranked = ranked[:num_samples]
        selected_indices = [s.index for s in selected_ranked]
        
        return selected_indices, selected_ranked


class BudgetOptimizer:
    """Optimize labeling budget allocation across categories/time."""
    
    def __init__(self):
        pass
    
    def allocate_budget(
        self,
        total_budget: int,
        category_sizes: Dict[str, int],
        category_uncertainty: Dict[str, float] = None,
        method: str = 'proportional'
    ) -> Dict[str, int]:
        """
        Allocate labeling budget across categories.
        
        Args:
            total_budget: Total labels available
            category_sizes: {category_name: num_samples}
            category_uncertainty: Optional {category_name: avg_uncertainty}
            method: 'proportional', 'uncertainty_weighted', 'equal'
        
        Returns:
            {category_name: num_labels_to_allocate}
        """
        categories = list(category_sizes.keys())
        sizes = np.array([category_sizes[c] for c in categories])
        
        if method == 'proportional':
            # Allocate proportional to category size
            weights = sizes / np.sum(sizes)
        elif method == 'uncertainty_weighted' and category_uncertainty:
            # Allocate more to uncertain categories
            uncertainties = np.array([category_uncertainty.get(c, 0.5) for c in categories])
            weights = uncertainties / np.sum(uncertainties)
        elif method == 'equal':
            # Equal allocation
            weights = np.ones(len(categories)) / len(categories)
        else:
            weights = sizes / np.sum(sizes)
        
        # Allocate budget
        allocation = {}
        remaining_budget = total_budget
        
        for cat, weight in zip(categories, weights):
            if cat == categories[-1]:
                # Last category gets remaining
                allocation[cat] = remaining_budget
            else:
                cat_budget = int(np.round(total_budget * weight))
                allocation[cat] = min(cat_budget, category_sizes[cat])
                remaining_budget -= allocation[cat]
        
        return allocation
    
    def estimate_labeling_cost(
        self,
        num_labels: int,
        cost_per_label: float = 2.0,
        time_per_label: float = 60  # seconds
    ) -> Dict:
        """
        Estimate cost and time for labeling.
        
        Args:
            num_labels: Number of labels to create
            cost_per_label: Dollar cost per label
            time_per_label: Seconds per label
        
        Returns:
            Cost and time estimates
        """
        total_cost = num_labels * cost_per_label
        total_time_seconds = num_labels * time_per_label
        total_time_hours = total_time_seconds / 3600
        total_time_days = total_time_hours / 8  # Assume 8-hour workday
        
        return {
            'num_labels': num_labels,
            'total_cost': float(total_cost),
            'total_time_seconds': total_time_seconds,
            'total_time_hours': float(total_time_hours),
            'total_time_days': float(total_time_days),
            'annotators_needed': max(1, int(np.ceil(total_time_days / 5)))  # 5-day workweek
        }


class AcquisitionFunction:
    """
    Different acquisition functions for sample selection.
    
    This is like a "recommendation scoring function" that rates
    how valuable each sample is to label.
    """
    
    @staticmethod
    def entropy_weighted(uncertainty: float, diversity: float, alpha: float = 0.5) -> float:
        """
        Entropy-weighted acquisition.
        
        AF = α * uncertainty + (1-α) * diversity
        """
        return alpha * uncertainty + (1 - alpha) * diversity
    
    @staticmethod
    def product_af(uncertainty: float, diversity: float) -> float:
        """
        Product-based acquisition.
        
        AF = uncertainty * diversity
        More selective: only high in BOTH.
        """
        return uncertainty * diversity
    
    @staticmethod
    def cost_sensitive_af(
        uncertainty: float,
        diversity: float,
        cost_per_label: float = 1.0,
        budget: float = 500.0
    ) -> float:
        """
        Cost-sensitive acquisition.
        
        Balances information gain with budget constraints.
        AF = (uncertainty * diversity) / cost_per_label
        """
        return (uncertainty * diversity) / max(cost_per_label, 1e-6)


def generate_ranking_report(
    ranked_samples: List[RankedSample],
    top_k: int = 20
) -> str:
    """Generate human-readable ranking report."""
    report = "Top-K Samples for Labeling\n"
    report += "=" * 80 + "\n"
    report += f"{'Rank':<6} {'Sample ID':<30} {'Uncertainty':<15} {'Diversity':<15} {'Score':<10}\n"
    report += "-" * 80 + "\n"
    
    for sample in ranked_samples[:top_k]:
        report += f"{sample.rank:<6} {sample.sample_id:<30} {sample.uncertainty:<15.4f} {sample.diversity:<15.4f} {sample.combined_score:<10.4f}\n"
    
    report += "-" * 80 + "\n"
    report += f"Total ranked: {len(ranked_samples)}\n"
    
    return report


if __name__ == "__main__":
    print("Ranking Strategy Module\n")
    
    # Simulate data
    num_samples = 1000
    sample_ids = [f"sample_{i:06d}" for i in range(num_samples)]
    uncertainty = np.random.rand(num_samples)
    embeddings = np.random.randn(num_samples, 256)
    labeled = [0, 1, 2, 5, 10]  # Already labeled
    
    # Rank samples
    ranking = RankingStrategy(uncertainty_weight=0.6, diversity_weight=0.4)
    ranked_samples = ranking.rank_samples(sample_ids, uncertainty, embeddings, labeled)
    
    # Get top-20
    selected_indices, top_ranked = ranking.select_batch(
        sample_ids, uncertainty, embeddings, 20, labeled
    )
    
    print("Top 20 ranked samples:")
    print(generate_ranking_report(top_ranked, top_k=20))
    
    # Budget optimization
    print("\n\nBudget Allocation:")
    optimizer = BudgetOptimizer()
    allocation = optimizer.allocate_budget(
        total_budget=500,
        category_sizes={
            'toxic': 2000,
            'harassment': 1000,
            'spam': 3000
        },
        method='proportional'
    )
    print(f"  {allocation}")
    
    # Cost estimation
    cost_est = optimizer.estimate_labeling_cost(
        num_labels=500,
        cost_per_label=2.0,
        time_per_label=60
    )
    print(f"\n\nCost Estimation (500 labels):")
    for key, value in cost_est.items():
        print(f"  {key}: {value}")
