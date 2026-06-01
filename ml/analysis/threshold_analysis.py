import json
import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure system logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

def run_probability_threshold_tuning() -> dict:
    """
    Evaluates classification performance across multiple probability thresholds
    to find the optimal balance between Recall and Precision for predictive maintenance.
    
    1. Loads test split data and serialization pickles
    2. Extracts probability scores for positive failure predictions
    3. Runs standard metrics evaluations at thresholds: [0.5, 0.4, 0.35, 0.3]
    4. Saves comparison results to ml/models/threshold_results.json
    5. Outputs formatted logs highlighting the best tuning threshold
    """
    logger.info("Initializing Probability Threshold Tuning Pipeline...")
    
    # 1. Load splits and optimal model
    X_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"
    best_model_path = MODELS_DIR / "best_model.pkl"
    
    if not (X_test_path.exists() and y_test_path.exists() and best_model_path.exists()):
        logger.error("Intermediate preprocessed test splits or serialized model are missing.")
        raise FileNotFoundError("Prerequisite modeling files are missing on disk.")
        
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    model = joblib.load(best_model_path)
    
    # 2. Extract probability of the positive class (failure condition)
    if not hasattr(model, "predict_proba"):
        logger.error("Model does not support probability forecasting (predict_proba).")
        raise AttributeError("Trained optimal model lacks predict_proba method.")
        
    logger.info("Extracting operational anomaly probability forecasting parameters...")
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # 3. Iterate through configurable threshold points
    thresholds = [0.5, 0.4, 0.35, 0.3]
    results = {}
    best_threshold = 0.5
    best_f1_score = -1.0
    
    logger.info("==================================================")
    logger.info("    PROBABILITY THRESHOLD OPTIMIZATION INDEX     ")
    logger.info("==================================================")
    
    for t in thresholds:
        # Predict binary classifications using threshold gating
        predictions = (probabilities > t).astype(int)
        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="binary", zero_division=0)
        rec = recall_score(y_test, predictions, average="binary", zero_division=0)
        f1 = f1_score(y_test, predictions, average="binary", zero_division=0)
        
        results[str(t)] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
        
        logger.info(f"Threshold: {t:.2f} -> Accuracy: {acc*100:.2f}%, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")
        
        # Track best threshold based on F1-score optimization
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = t
            
    logger.info("==================================================")
    logger.info(f"Optimal Threshold Selected: {best_threshold:.2f} with highest F1-Score of {best_f1_score:.4f}")
    logger.info(f"Optimal Metrics -> Precision: {results[str(best_threshold)]['precision']:.4f}, Recall: {results[str(best_threshold)]['recall']:.4f}")
    logger.info("==================================================")
    
    # 4. Save results metadata JSON to ml/models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = MODELS_DIR / "threshold_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": best_threshold,
            "threshold_metrics": results
        }, f, indent=4)
    logger.info(f"Successfully saved threshold results catalog metadata to: {results_path}")
    
    # 5. Save report text summary to outputs/reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "threshold_comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("    PREDICTIVE MAINTENANCE - THRESHOLD TUNING REPORT   \n")
        f.write("==================================================\n\n")
        f.write(f"Optimal Configured Threshold: {best_threshold}\n")
        f.write(f"Best Achieved F1-Score      : {best_f1_score:.4f}\n\n")
        f.write("Detailed Comparative Analysis:\n")
        f.write("--------------------------------------------------\n")
        for t_str, metrics in results.items():
            f.write(f"Gating Threshold: {t_str}\n")
            f.write(f"  - Accuracy : {metrics['accuracy'] * 100:.2f}%\n")
            f.write(f"  - Precision: {metrics['precision']:.4f}\n")
            f.write(f"  - Recall   : {metrics['recall']:.4f}\n")
            f.write(f"  - F1-Score : {metrics['f1_score']:.4f}\n\n")
            
    logger.info(f"Successfully saved threshold tuning comparison report to: {report_path}")
    
    return results

if __name__ == "__main__":
    logger.info("--- Standalone Threshold Optimization Pipeline Started ---")
    run_probability_threshold_tuning()
    logger.info("--- Standalone Threshold Optimization Pipeline Completed ---")
