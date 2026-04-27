"""
Comprehensive evaluation metrics for content moderation.

Metrics:
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Confusion Matrix
- Per-class metrics
- Fairness metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
from typing import Dict, Tuple
import json


class EvaluationMetrics:
    """Compute comprehensive metrics."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None
    ) -> Dict:
        """
        Compute all metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Prediction scores (for ROC-AUC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # ROC-AUC (if scores available)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['confusion_matrix'] = {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        
        # Additional metrics
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        
        return metrics
    
    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute per-class metrics."""
        num_classes = len(np.unique(y_true))
        
        metrics = {}
        for class_id in range(num_classes):
            y_true_binary = (y_true == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)
            
            metrics[f'class_{class_id}'] = EvaluationMetrics.compute_metrics(
                y_true_binary, y_pred_binary
            )
        
        return metrics
    
    @staticmethod
    def compute_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray
    ) -> Dict:
        """
        Compute fairness metrics across groups.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            groups: Group assignment for each sample
        """
        unique_groups = np.unique(groups)
        fairness = {}
        
        for group in unique_groups:
            group_mask = groups == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            if len(y_true_group) > 0:
                fairness[f'group_{group}'] = {
                    'accuracy': float(accuracy_score(y_true_group, y_pred_group)),
                    'recall': float(recall_score(y_true_group, y_pred_group, zero_division=0)),
                    'precision': float(precision_score(y_true_group, y_pred_group, zero_division=0)),
                    'num_samples': len(y_true_group)
                }
        
        # Compute disparities
        fairness['disparities'] = {}
        accuracies = [fairness[f'group_{g}']['accuracy'] for g in unique_groups]
        fairness['disparities']['accuracy_std'] = float(np.std(accuracies))
        fairness['disparities']['accuracy_max_diff'] = float(max(accuracies) - min(accuracies))
        
        return fairness
    
    @staticmethod
    def compute_rare_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        rare_class_label: int = 1
    ) -> Dict:
        """
        Focus on rare class (important for imbalanced data).
        
        For content moderation, unsafe content (class 1) is rare and critical.
        """
        rare_mask = y_true == rare_class_label
        
        rare_metrics = {
            'total_rare': int(np.sum(rare_mask)),
            'detected_rare': int(np.sum((y_pred == rare_class_label) & rare_mask)),
            'false_positives': int(np.sum((y_pred == rare_class_label) & ~rare_mask)),
            'false_negatives': int(np.sum((y_pred != rare_class_label) & rare_mask)),
        }
        
        if rare_metrics['total_rare'] > 0:
            rare_metrics['rare_recall'] = (
                rare_metrics['detected_rare'] / rare_metrics['total_rare']
            )
        
        if (rare_metrics['detected_rare'] + rare_metrics['false_positives']) > 0:
            rare_metrics['rare_precision'] = (
                rare_metrics['detected_rare'] / 
                (rare_metrics['detected_rare'] + rare_metrics['false_positives'])
            )
        
        return rare_metrics
    
    @staticmethod
    def format_metrics_report(metrics: Dict) -> str:
        """Format metrics as readable report."""
        report = "EVALUATION METRICS\n"
        report += "=" * 60 + "\n"
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                report += f"\n{key}:\n"
                for k, v in value.items():
                    report += f"  {k}: {v}\n"
            else:
                report += f"{key}: {value}\n"
        
        return report


class RareClassAnalysis:
    """Focused analysis on rare/harmful content."""
    
    @staticmethod
    def analyze_harmful_content(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidences: np.ndarray = None
    ) -> Dict:
        """
        Detailed analysis of harmful content detection.
        
        Critical for content moderation: failing to detect harmful content
        is worse than false positives.
        """
        analysis = {}
        
        # True positives (correctly caught harmful)
        tp_mask = (y_true == 1) & (y_pred == 1)
        analysis['true_positives'] = int(np.sum(tp_mask))
        
        # False negatives (missed harmful - CRITICAL)
        fn_mask = (y_true == 1) & (y_pred == 0)
        analysis['false_negatives'] = int(np.sum(fn_mask))
        analysis['false_negative_rate'] = float(
            np.sum(fn_mask) / max(1, np.sum(y_true == 1))
        )
        
        # False positives (over-flagged safe)
        fp_mask = (y_true == 0) & (y_pred == 1)
        analysis['false_positives'] = int(np.sum(fp_mask))
        analysis['false_positive_rate'] = float(
            np.sum(fp_mask) / max(1, np.sum(y_true == 0))
        )
        
        # Recall on harmful (ability to catch harmful)
        analysis['recall_harmful'] = float(
            analysis['true_positives'] / max(1, analysis['true_positives'] + analysis['false_negatives'])
        )
        
        # Specificity (ability to accept safe)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        analysis['specificity'] = float(
            tn / max(1, tn + fp_mask.sum())
        )
        
        # Confidence analysis
        if confidences is not None:
            fn_confidences = confidences[fn_mask]
            if len(fn_confidences) > 0:
                analysis['missed_harmful_avg_confidence'] = float(np.mean(fn_confidences))
                analysis['missed_harmful_max_confidence'] = float(np.max(fn_confidences))
                analysis['missed_harmful_min_confidence'] = float(np.min(fn_confidences))
        
        return analysis


class LearningCurveAnalysis:
    """Analyze learning efficiency."""
    
    @staticmethod
    def compute_sample_efficiency(
        accuracies: list,
        labels_used: list
    ) -> Dict:
        """
        Compute how efficiently labels are used.
        
        Key metric: accuracy gain per label.
        """
        if len(accuracies) < 2:
            return {}
        
        accuracy_gains = np.diff(accuracies)
        label_increases = np.diff(labels_used)
        
        efficiency = []
        for gain, labels in zip(accuracy_gains, label_increases):
            if labels > 0:
                efficiency.append(gain / labels)
        
        return {
            'mean_accuracy_per_label': float(np.mean(efficiency)),
            'std_accuracy_per_label': float(np.std(efficiency)),
            'total_accuracy_gain': float(accuracies[-1] - accuracies[0]),
            'total_labels_used': int(labels_used[-1])
        }
    
    @staticmethod
    def estimate_convergence(
        accuracies: list,
        target_accuracy: float = 0.90
    ) -> Dict:
        """Estimate when target accuracy is reached."""
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                return {
                    'target_reached': True,
                    'rounds_to_target': i + 1,
                    'accuracy_at_target': float(acc)
                }
        
        return {
            'target_reached': False,
            'rounds_to_target': None,
            'current_max_accuracy': float(max(accuracies)) if accuracies else 0
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate predictions
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.9, 0.85, 0.15, 0.4, 0.2, 0.95, 0.88, 0.1, 0.05, 0.9, 0.7, 0.8])
    
    # Compute metrics
    metrics = EvaluationMetrics.compute_metrics(y_true, y_pred, y_scores)
    
    print("Standard Metrics:")
    print(EvaluationMetrics.format_metrics_report(metrics))
    
    # Rare class analysis
    rare_metrics = RareClassAnalysis.analyze_harmful_content(y_true, y_pred, y_scores)
    print("\nRare Class (Harmful Content) Analysis:")
    print(EvaluationMetrics.format_metrics_report(rare_metrics))
    
    # Learning curves
    accuracies = [0.65, 0.72, 0.78, 0.83, 0.86]
    labels_used = [100, 200, 300, 400, 500]
    
    efficiency = LearningCurveAnalysis.compute_sample_efficiency(accuracies, labels_used)
    print("\nSample Efficiency:")
    print(EvaluationMetrics.format_metrics_report(efficiency))
    
    convergence = LearningCurveAnalysis.estimate_convergence(accuracies, target_accuracy=0.85)
    print("\nConvergence Analysis:")
    print(EvaluationMetrics.format_metrics_report(convergence))
