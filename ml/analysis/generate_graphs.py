import time
import logging
import matplotlib
# Prevent GUI blocking by setting non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List

# Configure logger
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
GRAPHS_DIR = BASE_DIR / "outputs" / "graphs"

def get_timestamped_filename(base_name: str) -> str:
    """Generates a timestamped filename to prevent file overwrites."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.png"

def plot_confusion_matrix(cm: np.ndarray, save_dir: Path = GRAPHS_DIR) -> Path:
    """Plots and saves a professional confusion matrix heatmap."""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = get_timestamped_filename("confusion_matrix")
    save_path = save_dir / filename
    fixed_path = save_dir / "confusion_matrix_heatmap.png"
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Safe', 'Failure'], yticklabels=['Safe', 'Failure'])
    plt.title("Confusion Matrix Heatmap", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Predicted Class Label")
    plt.ylabel("True Class Label")
    plt.tight_layout()
    
    # Save timestamped copy for historical/PDF report generators
    plt.savefig(save_path, dpi=150)
    # Save fixed copy for dashboard visual display
    plt.savefig(fixed_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved confusion matrix heatmap to: {save_path}")
    logger.info(f"Saved fixed confusion matrix heatmap to: {fixed_path}")
    return save_path

def plot_model_comparisons(metrics: Dict[str, Dict[str, float]], save_dir: Path = GRAPHS_DIR) -> None:
    """Plots and saves classification validation F1-score and accuracy comparison bar charts."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(metrics.keys())
    accuracies = [m["accuracy"] * 100 for m in metrics.values()]
    f1_scores = [m["f1_score"] for m in metrics.values()]
    
    # 1. Save F1-Score bar comparison chart
    f1_filename = get_timestamped_filename("model_f1_comparison")
    f1_path = save_dir / f1_filename
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=f1_scores, palette="viridis")
    plt.title("Model Comparative validation F1-Scores", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Trained Classifier")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1.05)
    
    # Annotate bar values
    for i, score in enumerate(f1_scores):
        plt.text(i, score + 0.02, f"{score:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f1_path, dpi=150)
    plt.close()
    logger.info(f"Saved F1-score model comparative chart to: {f1_path}")
    
    # 2. Save Accuracy bar comparison chart
    acc_filename = get_timestamped_filename("model_accuracy_comparison")
    acc_path = save_dir / acc_filename
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=accuracies, palette="magma")
    plt.title("Model Comparative validation Accuracies (%)", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Trained Classifier")
    plt.ylabel("Validation Accuracy (%)")
    plt.ylim(0, 105)
    
    # Annotate bar values
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1.0, f"{acc:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()
    logger.info(f"Saved accuracy model comparative chart to: {acc_path}")

def plot_feature_importances(importances: List[float], feature_names: List[str], save_dir: Path = GRAPHS_DIR) -> Path:
    """Plots and saves feature importance horizontal bar chart."""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = get_timestamped_filename("feature_importance")
    save_path = save_dir / filename
    
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="crest")
    plt.title("Tuned Optimal Model - Feature Importance Profile", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Relative Feature Importance Score")
    plt.ylabel("Input Features")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance chart to: {save_path}")
    return save_path
