"""
Diversity sampling for active learning.

Key idea: Don't just pick uncertain samples. Pick uncertain samples
that are DIFFERENT from each other to maximize information gain.

Uses clustering and embedding distance.
"""

import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import torch


class DiversitySampler:
    """Diversity-based sampling using k-center greedy."""
    
    def __init__(self, method: str = 'k_center'):
        """
        Args:
            method: 'k_center' (greedy k-center problem)
        """
        self.method = method
    
    def select_batch_greedy(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        exclude_indices: List[int] = None
    ) -> List[int]:
        """
        Greedy k-center selection.
        
        Minimizes the maximum distance from any point to selected centers.
        Key insight: selects diverse samples that spread out in embedding space.
        
        Args:
            embeddings: [num_samples, embedding_dim]
            num_samples: How many to select
            exclude_indices: Indices to exclude
        
        Returns:
            Indices of selected samples
        """
        num_total = embeddings.shape[0]
        
        # Initialize: exclude labeled samples
        available_mask = np.ones(num_total, dtype=bool)
        if exclude_indices:
            available_mask[exclude_indices] = False
        
        available_indices = np.where(available_mask)[0]
        
        # Start with a random sample
        selected_indices = [np.random.choice(available_indices)]
        
        # Iteratively select next center
        for _ in range(num_samples - 1):
            if len(selected_indices) == 0:
                break
            
            # Compute distance from each unselected point to nearest selected point
            selected_embeddings = embeddings[selected_indices]
            distances = cdist(embeddings[available_mask], selected_embeddings, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            
            # Select point that maximizes minimum distance (farthest from centers)
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(available_indices[farthest_idx])
            
            # Remove from available
            available_indices = np.delete(available_indices, farthest_idx)
        
        return selected_indices
    
    def select_batch_clustering(
        self,
        embeddings: np.ndarray,
        num_samples: int,
        exclude_indices: List[int] = None,
        num_clusters: int = None
    ) -> List[int]:
        """
        Clustering-based diversity selection.
        
        1. Cluster embeddings into k clusters
        2. Select top samples from each cluster by uncertainty/score
        
        Args:
            embeddings: [num_samples, embedding_dim]
            num_samples: How many to select
            exclude_indices: Indices to exclude
            num_clusters: Number of clusters (default: sqrt(num_samples))
        
        Returns:
            Indices of selected samples
        """
        num_total = embeddings.shape[0]
        if num_clusters is None:
            num_clusters = max(2, int(np.sqrt(num_samples)))
        
        # Cluster
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Mark excluded as unavailable
        available_mask = np.ones(num_total, dtype=bool)
        if exclude_indices:
            available_mask[exclude_indices] = False
        
        # Select from each cluster
        selected = []
        samples_per_cluster = num_samples // num_clusters
        remainder = num_samples % num_clusters
        
        for cluster_id in range(num_clusters):
            cluster_mask = (cluster_labels == cluster_id) & available_mask
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Select random samples from this cluster
            num_to_select = samples_per_cluster + (1 if cluster_id < remainder else 0)
            num_to_select = min(num_to_select, len(cluster_indices))
            
            cluster_selected = np.random.choice(cluster_indices, size=num_to_select, replace=False)
            selected.extend(cluster_selected.tolist())
        
        return selected[:num_samples]


class HybridSampler:
    """
    Hybrid active learning combining uncertainty and diversity.
    
    Formula: score = uncertainty_weight * uncertainty + diversity_weight * diversity
    """
    
    def __init__(self, uncertainty_weight: float = 0.5, diversity_weight: float = 0.5):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
    
    def compute_diversity_scores(
        self,
        embeddings: np.ndarray,
        exclude_indices: List[int] = None
    ) -> np.ndarray:
        """
        Compute diversity scores (higher = more unique).
        
        Uses distance to nearest neighbor as diversity metric.
        """
        num_total = embeddings.shape[0]
        
        # Initialize score
        diversity_scores = np.zeros(num_total)
        
        # Mark excluded
        available_mask = np.ones(num_total, dtype=bool)
        if exclude_indices:
            available_mask[exclude_indices] = False
        
        # Compute average distance to K nearest neighbors
        k_neighbors = min(5, num_total // 10)
        
        for i in range(num_total):
            if not available_mask[i]:
                continue
            
            # Distance to all other available points
            distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
            distances[i] = np.inf
            distances[~available_mask] = np.inf
            
            # Average distance to k nearest
            if np.sum(np.isfinite(distances)) >= k_neighbors:
                nearest_distances = np.sort(distances)[:k_neighbors]
                diversity_scores[i] = np.mean(nearest_distances)
        
        # Normalize
        diversity_scores = (diversity_scores - np.min(diversity_scores)) / (np.max(diversity_scores) - np.min(diversity_scores) + 1e-8)
        
        return diversity_scores
    
    def select_batch(
        self,
        embeddings: np.ndarray,
        uncertainty_scores: np.ndarray,
        num_samples: int,
        exclude_indices: List[int] = None
    ) -> List[int]:
        """
        Select batch using combined uncertainty + diversity.
        
        Args:
            embeddings: [num_samples, embedding_dim]
            uncertainty_scores: [num_samples] uncertainty from model
            num_samples: How many to select
            exclude_indices: Indices to exclude
        
        Returns:
            Indices of selected samples
        """
        # Normalize uncertainty scores
        uncertainty_norm = (uncertainty_scores - np.min(uncertainty_scores)) / (np.max(uncertainty_scores) - np.min(uncertainty_scores) + 1e-8)
        
        # Compute diversity scores
        diversity_scores = self.compute_diversity_scores(embeddings, exclude_indices)
        
        # Combine
        combined_scores = (
            self.uncertainty_weight * uncertainty_norm +
            self.diversity_weight * diversity_scores
        )
        
        # Exclude labeled
        if exclude_indices:
            combined_scores[exclude_indices] = -np.inf
        
        # Select top-k
        selected = np.argsort(combined_scores)[::-1][:num_samples]
        
        return selected.tolist()


def compute_embedding(
    texts: List[str],
    images: List,
    text_model,
    image_model
) -> np.ndarray:
    """
    Compute multimodal embeddings for diversity sampling.
    
    Args:
        texts: List of text strings
        images: List of PIL images
        text_model: Text encoder (returns embeddings)
        image_model: Image encoder (returns embeddings)
    
    Returns:
        [num_samples, embedding_dim] combined embeddings
    """
    # This is a placeholder. In practice, use CLIP embeddings or fine-tuned encoders
    # For now, simulate embeddings
    num_samples = len(texts)
    embedding_dim = 256
    
    embeddings = np.random.randn(num_samples, embedding_dim)
    return embeddings


def analyze_diversity(
    embeddings: np.ndarray,
    selected_indices: List[int]
) -> dict:
    """
    Analyze diversity of selected samples.
    
    Returns:
        Statistics on diversity
    """
    selected_embeddings = embeddings[selected_indices]
    
    # Pairwise distances
    distances = cdist(selected_embeddings, selected_embeddings, metric='euclidean')
    
    # Remove diagonal
    distances_no_diag = distances[~np.eye(distances.shape[0], dtype=bool)]
    
    stats = {
        'mean_distance': float(np.mean(distances_no_diag)),
        'min_distance': float(np.min(distances_no_diag)),
        'max_distance': float(np.max(distances_no_diag)),
        'std_distance': float(np.std(distances_no_diag)),
    }
    
    return stats


if __name__ == "__main__":
    print("Diversity Sampling Module\n")
    
    # Simulate embeddings
    num_samples = 1000
    embedding_dim = 256
    embeddings = np.random.randn(num_samples, embedding_dim)
    
    # Greedy k-center
    print("K-Center Greedy:")
    sampler = DiversitySampler()
    selected_greedy = sampler.select_batch_greedy(embeddings, num_samples=100)
    print(f"  Selected: {len(selected_greedy)} samples")
    
    diversity_stats = analyze_diversity(embeddings, selected_greedy)
    print(f"  Diversity stats: {diversity_stats}")
    print()
    
    # Clustering-based
    print("Clustering-based:")
    selected_clustering = sampler.select_batch_clustering(embeddings, num_samples=100)
    print(f"  Selected: {len(selected_clustering)} samples")
    print()
    
    # Hybrid
    print("Hybrid (Uncertainty + Diversity):")
    uncertainty_scores = np.random.rand(num_samples)
    hybrid_sampler = HybridSampler(uncertainty_weight=0.5, diversity_weight=0.5)
    selected_hybrid = hybrid_sampler.select_batch(embeddings, uncertainty_scores, num_samples=100)
    print(f"  Selected: {len(selected_hybrid)} samples")
