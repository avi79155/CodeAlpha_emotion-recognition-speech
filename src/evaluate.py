"""
Model Evaluation Module

This module provides evaluation metrics and visualization for trained models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model performance using various metrics and visualizations.
    """

    def __init__(self, class_names: List[str]):
        """
        Initialize ModelEvaluator.

        Args:
            class_names (List[str]): List of emotion class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        logger.info(f"ModelEvaluator initialized with {self.num_classes} classes")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        logger.info(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        return metrics

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            np.ndarray: Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion matrix shape: {cm.shape}")
        return cm

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            str: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        logger.info("Classification report generated")
        return report

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             figsize: Tuple[int, int] = (10, 8),
                             save_path: str = None) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.show()

    def plot_training_history(self, history: Dict[str, List[float]],
                             figsize: Tuple[int, int] = (14, 5),
                             save_path: str = None) -> None:
        """
        Plot training and validation metrics.

        Args:
            history (Dict): Training history dictionary
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Accuracy plot
        axes[0].plot(history.get('accuracy', []), label='Training Accuracy', linewidth=2)
        axes[0].plot(history.get('val_accuracy', []), label='Validation Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.get('loss', []), label='Training Loss', linewidth=2)
        axes[1].plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        plt.show()

    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: str = None) -> None:
        """
        Compare metrics across different models.

        Args:
            metrics_dict (Dict): Dictionary with model names and their metrics
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save figure
        """
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(metrics_names))
        width = 0.15

        plt.figure(figsize=figsize)

        for i, (model_name, metrics) in enumerate(metrics_dict.items()):
            values = [metrics.get(metric, 0) for metric in metrics_names]
            plt.bar(x + i * width, values, width, label=model_name)

        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width * (len(metrics_dict) - 1) / 2, metrics_names)
        plt.legend(fontsize=10)
        plt.ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to {save_path}")
        plt.show()

    def plot_class_distribution(self, y: np.ndarray,
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: str = None) -> None:
        """
        Plot distribution of emotion classes.

        Args:
            y (np.ndarray): Label array
            figsize (Tuple[int, int]): Figure size
            save_path (str): Path to save figure
        """
        unique, counts = np.unique(y, return_counts=True)
        plt.figure(figsize=figsize)
        bars = plt.bar(self.class_names, counts, color='skyblue', edgecolor='black')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Emotion Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    logger.info("Evaluation module loaded successfully")
