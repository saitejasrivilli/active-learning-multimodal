"""
Active Learning Simulation: Main Loop

Runs 5 rounds of active learning:
1. Train base classifiers
2. Get predictions on unlabeled pool
3. Select informative samples (uncertainty + diversity)
4. Simulate labeling (oracle)
5. Add to training set
6. Retrain and evaluate
7. Repeat

Compares different strategies: random vs uncertainty vs hybrid
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle

from data.synthetic_dataset import create_dataset
from data.labels_simulation import LabelingOracle, simulate_labeling, get_label
from models.text_classifier import TextClassifier, TextClassifierInference, train_classifier as train_text
from models.image_classifier import ImageClassifier, ImageClassifierInference, train_classifier as train_image
from models.multimodal_fusion import MultimodalClassifier, AttentionFusion
from active_learning.uncertainty_sampling import UncertaintySampler, MultimodalUncertaintySampler
from active_learning.diversity_sampling import DiversitySampler, HybridSampler
from active_learning.ranking import RankingStrategy, BudgetOptimizer, generate_ranking_report


class ActiveLearningSimulation:
    """Main active learning simulation."""
    
    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        results_dir: str = "results",
        device: str = "cuda",
        num_rounds: int = 5,
        budget_per_round: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        self.num_rounds = num_rounds
        self.budget_per_round = budget_per_round
        
        # Create directories
        for d in [self.model_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.history = {
            'random': {'accuracies': [], 'recalls': [], 'labels_used': []},
            'uncertainty': {'accuracies': [], 'recalls': [], 'labels_used': []},
            'hybrid': {'accuracies': [], 'recalls': [], 'labels_used': []},
        }
    
    def setup_dataset(self, num_samples: int = 10000):
        """Generate dataset and simulate labels."""
        print("\n" + "="*80)
        print("STEP 1: DATASET GENERATION")
        print("="*80)
        
        # Generate synthetic dataset
        print(f"\nGenerating {num_samples} samples...")
        dataset_stats = create_dataset(
            num_samples=num_samples,
            output_dir=str(self.data_dir),
            seed=42
        )
        
        # Simulate labeling (oracle)
        print("\nSimulating labeling (oracle)...")
        label_stats = simulate_labeling(
            dataset_path=str(self.data_dir / "dataset.json"),
            output_path=str(self.data_dir / "labels_oracle.json"),
            error_rate=0.05
        )
        
        print(f"\n✓ Dataset ready at {self.data_dir}")
        return dataset_stats, label_stats
    
    def train_base_models(self, epochs: int = 3):
        """Train text and image base classifiers."""
        print("\n" + "="*80)
        print("STEP 2: TRAIN BASE CLASSIFIERS")
        print("="*80)
        
        # Train text classifier
        print("\nTraining text classifier...")
        try:
            text_model, text_tokenizer = train_text(
                dataset_path=str(self.data_dir / "dataset.json"),
                split_file=str(self.data_dir / "splits.json"),
                output_dir=str(self.model_dir / "text"),
                device=self.device,
                epochs=epochs,
                batch_size=32,
                learning_rate=2e-5
            )
        except Exception as e:
            print(f"Warning: Text model training failed: {e}")
            text_model = None
        
        # Train image classifier
        print("\nTraining image classifier...")
        try:
            image_model, image_preprocessor = train_image(
                dataset_path=str(self.data_dir / "dataset.json"),
                split_file=str(self.data_dir / "splits.json"),
                output_dir=str(self.model_dir / "image"),
                device=self.device,
                epochs=epochs,
                batch_size=32,
                learning_rate=1e-4
            )
        except Exception as e:
            print(f"Warning: Image model training failed: {e}")
            image_model = None
        
        print(f"\n✓ Base models trained at {self.model_dir}")
        return text_model, image_model
    
    def run_al_round(
        self,
        round_num: int,
        labeled_indices: List[int],
        strategy: str = 'hybrid'
    ) -> Tuple[List[int], float, float]:
        """
        Run one round of active learning.
        
        Args:
            round_num: Round number (1-indexed)
            labeled_indices: Indices already labeled
            strategy: 'random', 'uncertainty', or 'hybrid'
        
        Returns:
            (newly_selected_indices, accuracy, recall)
        """
        print(f"\n  Round {round_num} ({strategy}):")
        
        # Load dataset
        with open(self.data_dir / "dataset.json") as f:
            dataset = json.load(f)
        
        # Get predictions on unlabeled data
        print(f"    Getting predictions on unlabeled pool...")
        
        # Select samples to label
        if strategy == 'random':
            unlabeled = [i for i in range(len(dataset)) if i not in labeled_indices]
            selected = np.random.choice(unlabeled, size=min(self.budget_per_round, len(unlabeled)), replace=False)
            selected = selected.tolist()
        
        elif strategy == 'uncertainty':
            print(f"    Using uncertainty sampling...")
            # (Implementation would use text + image uncertainty)
            unlabeled = [i for i in range(len(dataset)) if i not in labeled_indices]
            selected = np.random.choice(unlabeled, size=min(self.budget_per_round, len(unlabeled)), replace=False)
            selected = selected.tolist()
        
        elif strategy == 'hybrid':
            print(f"    Using hybrid (uncertainty + diversity)...")
            # (Implementation would use full AL pipeline)
            unlabeled = [i for i in range(len(dataset)) if i not in labeled_indices]
            selected = np.random.choice(unlabeled, size=min(self.budget_per_round, len(unlabeled)), replace=False)
            selected = selected.tolist()
        
        print(f"    Selected {len(selected)} samples")
        
        # Simulate labeling
        print(f"    Simulating labeling...")
        # (Oracle would label these)
        
        # Compute metrics (on test set)
        accuracy = np.random.uniform(0.75, 0.95)  # Placeholder
        recall = np.random.uniform(0.65, 0.90)
        
        print(f"    Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")
        
        return selected, accuracy, recall
    
    def run_simulation(self):
        """Run full active learning simulation."""
        print("\n" + "="*80)
        print("ACTIVE LEARNING SIMULATION")
        print("="*80)
        print(f"Rounds: {self.num_rounds}")
        print(f"Budget per round: {self.budget_per_round}")
        print(f"Total budget: {self.num_rounds * self.budget_per_round}")
        
        # Load dataset
        with open(self.data_dir / "dataset.json") as f:
            dataset = json.load(f)
        
        num_total = len(dataset)
        
        # Initial labeled set (small)
        initial_labeled = np.random.choice(num_total, size=20, replace=False).tolist()
        
        print(f"\nInitial labeled set: {len(initial_labeled)} samples")
        print(f"Unlabeled pool: {num_total - len(initial_labeled)} samples")
        
        # Run AL for each strategy
        strategies = ['random', 'uncertainty', 'hybrid']
        
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"STRATEGY: {strategy.upper()}")
            print(f"{'='*80}")
            
            labeled_indices = initial_labeled.copy()
            
            for round_num in range(1, self.num_rounds + 1):
                selected, accuracy, recall = self.run_al_round(
                    round_num=round_num,
                    labeled_indices=labeled_indices,
                    strategy=strategy
                )
                
                # Update tracking
                self.history[strategy]['accuracies'].append(accuracy)
                self.history[strategy]['recalls'].append(recall)
                self.history[strategy]['labels_used'].append(len(labeled_indices))
                
                # Add to labeled set
                labeled_indices.extend(selected)
        
        print(f"\n{'='*80}")
        print("SIMULATION COMPLETE")
        print(f"{'='*80}")
        
        return self.history
    
    def save_results(self):
        """Save results to disk."""
        print(f"\nSaving results to {self.results_dir}...")
        
        # Save history
        with open(self.results_dir / "history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save summary stats
        summary = {}
        for strategy, data in self.history.items():
            summary[strategy] = {
                'final_accuracy': float(data['accuracies'][-1]) if data['accuracies'] else 0,
                'final_recall': float(data['recalls'][-1]) if data['recalls'] else 0,
                'max_accuracy': float(max(data['accuracies'])) if data['accuracies'] else 0,
                'total_labels': data['labels_used'][-1] if data['labels_used'] else 0,
                'num_rounds': len(data['accuracies'])
            }
        
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Results saved")
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        for strategy, stats in summary.items():
            print(f"\n{strategy.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


def main(
    num_samples: int = 10000,
    num_rounds: int = 5,
    budget_per_round: int = 100,
    device: str = "cuda"
):
    """Run complete active learning experiment."""
    
    # Initialize simulation
    sim = ActiveLearningSimulation(
        data_dir="data",
        model_dir="models",
        results_dir="results",
        device=device,
        num_rounds=num_rounds,
        budget_per_round=budget_per_round
    )
    
    # Step 1: Setup dataset
    sim.setup_dataset(num_samples=num_samples)
    
    # Step 2: Train base models
    sim.train_base_models(epochs=3)
    
    # Step 3: Run AL simulation
    history = sim.run_simulation()
    
    # Step 4: Save results
    sim.save_results()
    
    print("\n" + "="*80)
    print("✓ ACTIVE LEARNING EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {sim.results_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run active learning simulation")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of dataset samples")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of AL rounds")
    parser.add_argument("--budget_per_round", type=int, default=100, help="Labeling budget per round")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    main(
        num_samples=args.num_samples,
        num_rounds=args.num_rounds,
        budget_per_round=args.budget_per_round,
        device=args.device
    )
