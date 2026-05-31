import joblib
import pandas as pd
import logging
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Modular imports
from ml.analysis.generate_graphs import (
    plot_confusion_matrix, plot_model_comparisons, plot_feature_importances
)
from ml.analysis.generate_reports import save_classification_report_txt
from ml.analysis.statistical_analysis import perform_dataset_statistical_analysis

# Configure system logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

def run_evaluation_pipeline() -> None:
    """
    Executes the optimal classifier validation evaluation pipeline.
    
    1. Loads test splits and persisted best-tuned classifier model.
    2. Conducts dataset descriptive statistics.
    3. Runs inference on test features.
    4. Computes KPIs: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    5. Saves detailed classification report text to outputs/reports/ with timestamps.
    6. Plots and saves confusion matrix heatmaps to outputs/graphs/ with timestamps.
    7. Plots and saves model accuracy and F1 comparative charts to outputs/graphs/.
    8. Plots and saves optimal classifier feature importances chart if supported.
    """
    logger.info("Executing Optimal Classifier Evaluation Pipeline...")
    
    # 1. Check optimal model weights existence
    best_model_path = MODELS_DIR / "best_model.pkl"
    if not best_model_path.exists():
        logger.error(f"Best model pickle weights missing in: {best_model_path}. Train optimal models first.")
        raise FileNotFoundError(f"Best model weights missing at: {best_model_path}")
        
    # 2. Check processed splits existence
    if not (PROCESSED_DIR / "X_test.csv").exists():
        logger.error(f"Validation test splits missing in: {PROCESSED_DIR}. Run preprocessing pipeline first.")
        raise FileNotFoundError(f"Test splits missing at: {PROCESSED_DIR}")
        
    # 3. Load dataset splits and optimal model
    logger.info("Loading preprocessed validation test splits and model weights...")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    best_model = joblib.load(best_model_path)
    
    # 4. Perform Descriptive Statistical Checks
    perform_dataset_statistical_analysis(X_test)
    
    # 5. Run inference predictions
    logger.info("Running optimal model predictions on test split...")
    y_pred = best_model.predict(X_test)
    
    # 6. Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    
    class_report_str = classification_report(y_test, y_pred, target_names=["Safe", "Failure"])
    cm = confusion_matrix(y_test, y_pred)
    
    metrics_summary_str = (
        f"Model Accuracy : {acc * 100:.2f}%\n"
        f"Precision Score: {prec:.4f}\n"
        f"Recall Score   : {rec:.4f}\n"
        f"F1-Score       : {f1:.4f}\n"
    )
    
    logger.info(f"\n--- Validation Performance Metrics Summary ---\n{metrics_summary_str}")
    logger.info(f"\n--- Classification Report Matrix Table ---\n{class_report_str}")
    
    # 7. Write textual reports
    save_classification_report_txt(metrics_summary_str, class_report_str)
    
    # 8. Generate confusion matrix heatmap plot
    plot_confusion_matrix(cm)
    
    # 9. Load comparative metrics JSON to draw comparisons plots
    metrics_path = MODELS_DIR / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
        plot_model_comparisons(all_metrics)
    else:
        logger.warning(f"model_metrics.json not found in {MODELS_DIR}. Comparative plots will be bypassed.")
        
    # 10. Generate relative feature importance horizontal plot if optimal classifier supports it
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_.tolist()
        plot_feature_importances(importances, FEATURE_COLUMNS)
    else:
        logger.info("Optimal classifier does not export feature importances metrics. Plot bypassed.")

if __name__ == "__main__":
    logger.info("--- Standalone Model Evaluation Pipeline Started ---")
    run_evaluation_pipeline()
    logger.info("--- Standalone Model Evaluation Pipeline Completed Successfully ---")
