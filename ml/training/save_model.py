import json
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

# Configure local module logger
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "ml" / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

def save_best_model_and_metadata(
    model: Any,
    model_name: str,
    metrics: Dict[str, Dict[str, float]],
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR
) -> None:
    """
    Saves the best-performing model, metadata metrics, and a comparison report.
    
    Args:
        model (Any): The fitted model instance to save.
        model_name (str): The name of the best-performing model.
        metrics (Dict): Performance results dictionary for all models.
        models_dir (Path): Output directory for model files.
        reports_dir (Path): Output directory for textual metrics reports.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save optimal model weights
    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(model, best_model_path)
    logger.info(f"Successfully saved optimal model weights to: {best_model_path}")
    
    # 2. Save model metrics as JSON for easy downstream program access
    metrics_path = models_dir / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Successfully saved metrics metadata JSON to: {metrics_path}")
    
    # 3. Save comparative model report text summary
    report_path = reports_dir / "model_comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("  PREDICTIVE MAINTENANCE - MODEL TRAINING COMPARISON REPORT  \n")
        f.write("==================================================\n\n")
        f.write(f"Selected Optimal Model: {model_name}\n\n")
        f.write("Comparative Model Metrics:\n")
        f.write("--------------------------------------------------\n")
        for name, scores in metrics.items():
            f.write(f"Model Name: {name}\n")
            f.write(f"  - Validation Accuracy: {scores['accuracy'] * 100:.2f}%\n")
            f.write(f"  - Validation F1-Score: {scores['f1_score']:.4f}\n\n")
            
    logger.info(f"Successfully saved comparison report text summary to: {report_path}")
