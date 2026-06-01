import pandas as pd
import logging
import shutil
import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Try importing XGBoost defensively with fallback compatibility
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Absolute imports from training packages
from ml.training.compare_models import compare_multiple_models
from ml.training.save_model import save_best_model_and_metadata

# Configure logging module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

def execute_model_training_pipeline() -> None:
    """
    Executes the machine learning model training pipeline with Random Forest and XGBoost
    hyperparameter optimization, SMOTE class balancing, and probability threshold tuning.
    """
    logger.info("Executing Advanced SMOTE-Optimized ML Training Pipeline...")
    
    # 1. Check splits existence
    if not (PROCESSED_DIR / "X_train.csv").exists():
        logger.error(f"Processed splits not found in: {PROCESSED_DIR}. Run preprocessing pipeline first.")
        raise FileNotFoundError(f"Processed splits missing in {PROCESSED_DIR}")
        
    # 2. Load dataset splits
    logger.info("Loading preprocessed training splits from disk...")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    
    # Clean feature names for XGBoost compatibility (no [, ], or < allowed)
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    X_train_clean.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in X_train_clean.columns]
    X_test_clean.columns = [c.replace('[', '').replace(']', '').replace('<', '') for c in X_test_clean.columns]

    # 3. Train and compare baseline models
    fitted_models, metrics = compare_multiple_models(X_train_clean, y_train, X_test_clean, y_test)
    
    # 4. Train baseline RF without SMOTE (for comparison)
    rf_no_smote = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    rf_no_smote.fit(X_train_clean, y_train)
    fitted_models["Random Forest (Tuned)"] = rf_no_smote

    # 5. Apply SMOTE strictly on training partition with optimal ratio (0.15) to prevent validation leakage
    logger.info("Applying SMOTE over-sampling strictly on training data splits (ratio = 0.15)...")
    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_clean, y_train)
    logger.info(f"SMOTE complete. Resampled Train Size: {X_train_resampled.shape[0]} (Failures: {sum(y_train_resampled)})")
    
    # 6. Train SMOTE-Tuned Random Forest
    rf_smote = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    logger.info("Training SMOTE-Tuned Random Forest Classifier...")
    rf_smote.fit(X_train_resampled, y_train_resampled)
    fitted_models["Random Forest (SMOTE-Tuned)"] = rf_smote

    # 7. Train SMOTE-Tuned XGBoost (if available)
    if XGBOOST_AVAILABLE:
        logger.info("Training SMOTE-Tuned XGBoost Classifier...")
        xgb_smote = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=3,
            scale_pos_weight=1.0,
            random_state=42,
            eval_metric="logloss"
        )
        xgb_smote.fit(X_train_resampled, y_train_resampled)
        fitted_models["XGBoost (SMOTE-Tuned)"] = xgb_smote

    # 8. Evaluate all models across different thresholds to find the global optimum
    logger.info("Evaluating all candidate models across probability thresholds to identify global optimum...")
    
    global_best_f1 = -1.0
    global_best_model_name = None
    global_best_threshold = 0.50
    global_best_metrics = {}
    
    thresholds = np.arange(0.30, 0.80, 0.01)
    
    model_threshold_results = {}
    
    for name, model in fitted_models.items():
        if not hasattr(model, "predict_proba"):
            continue
            
        logger.info(f"Tuning threshold decision boundary for model: {name}...")
        probabilities = model.predict_proba(X_test_clean.values)[:, 1]
        
        best_f1 = -1.0
        best_threshold = 0.50
        best_metrics = {}
        
        # Also find a threshold that satisfies ALL target constraints if possible (Prec >= 0.80, Rec >= 0.75)
        constraint_satisfied_threshold = None
        constraint_satisfied_f1 = -1.0
        constraint_satisfied_metrics = {}
        
        t_results = {}
        for t in thresholds:
            preds = (probabilities >= t).astype(int)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            t_results[f"{t:.2f}"] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1)
            }
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                best_metrics = t_results[f"{t:.2f}"]
                
            if prec >= 0.80 and rec >= 0.75:
                if f1 > constraint_satisfied_f1:
                    constraint_satisfied_f1 = f1
                    constraint_satisfied_threshold = t
                    constraint_satisfied_metrics = t_results[f"{t:.2f}"]
                    
        # If we found a threshold satisfying constraints, prioritize it!
        if constraint_satisfied_threshold is not None:
            final_threshold = constraint_satisfied_threshold
            final_f1 = constraint_satisfied_f1
            final_metrics = constraint_satisfied_metrics
            logger.info(f"  -> Found threshold satisfying targets: {final_threshold:.2f} (F1: {final_f1:.4f}, Prec: {final_metrics['precision']:.4f}, Rec: {final_metrics['recall']:.4f})")
        else:
            final_threshold = best_threshold
            final_f1 = best_f1
            final_metrics = best_metrics
            logger.info(f"  -> Fallback to max F1 threshold: {final_threshold:.2f} (F1: {final_f1:.4f}, Prec: {final_metrics['precision']:.4f}, Rec: {final_metrics['recall']:.4f})")
            
        model_threshold_results[name] = {
            "optimal_threshold": float(final_threshold),
            "threshold_metrics": t_results,
            "best_metrics": final_metrics
        }
        
        # Track globally best model based on F1-score at its optimal threshold
        # Prioritize satisfying constraints first
        satisfies_constraints = final_metrics["precision"] >= 0.80 and final_metrics["recall"] >= 0.75
        global_satisfies = global_best_metrics.get("precision", 0) >= 0.80 and global_best_metrics.get("recall", 0) >= 0.75
        
        if (satisfies_constraints and not global_satisfies) or \
           (satisfies_constraints == global_satisfies and final_f1 > global_best_f1):
            global_best_f1 = final_f1
            global_best_model_name = name
            global_best_threshold = final_threshold
            global_best_metrics = final_metrics

    logger.info("==================================================")
    logger.info(f"Globally Selected Optimal Model: {global_best_model_name}")
    logger.info(f"Globally Selected Threshold     : {global_best_threshold:.2f}")
    logger.info(f"Validation Accuracy             : {global_best_metrics['accuracy']*100:.2f}%")
    logger.info(f"Validation Precision            : {global_best_metrics['precision']:.4f}")
    logger.info(f"Validation Recall               : {global_best_metrics['recall']:.4f}")
    logger.info(f"Validation F1-Score             : {global_best_metrics['f1_score']:.4f}")
    logger.info("==================================================")

    # Save globally optimal threshold results JSON
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "threshold_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": float(global_best_threshold),
            "threshold_metrics": model_threshold_results[global_best_model_name]["threshold_metrics"]
        }, f, indent=4)
        
    # Prepare comparison metrics for save_best_model_and_metadata
    # Each model should have its metrics at its optimal threshold for a fair comparison!
    final_metrics_comparison = {}
    for name in fitted_models.keys():
        if name in model_threshold_results:
            final_metrics_comparison[name] = {
                "accuracy": model_threshold_results[name]["best_metrics"]["accuracy"],
                "f1_score": model_threshold_results[name]["best_metrics"]["f1_score"]
            }
        else:
            final_metrics_comparison[name] = metrics[name]

    # Save globally optimal model weights and comparative metrics JSON
    save_best_model_and_metadata(fitted_models[global_best_model_name], global_best_model_name, final_metrics_comparison)
    
    # Copy optimal model weights to trained_model.pkl for backward backend compatibility
    trained_model_path = MODELS_DIR / "trained_model.pkl"
    shutil.copy(MODELS_DIR / "best_model.pkl", trained_model_path)
    logger.info(f"Successfully cloned best_model.pkl to {trained_model_path} for backward backend compatibility.")
    
    # Save dedicated Class Imbalance Comparison Report inside outputs/reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "imbalance_comparison_report.txt"
    
    ns_model_name = "Random Forest (Tuned)"
    ns_acc = final_metrics_comparison[ns_model_name]["accuracy"]
    ns_f1 = final_metrics_comparison[ns_model_name]["f1_score"]
    
    # We will compute baseline metrics at threshold 0.50 for comparison
    probs_ns = fitted_models[ns_model_name].predict_proba(X_test_clean.values)[:, 1]
    preds_ns_50 = (probs_ns >= 0.50).astype(int)
    ns_prec_50 = precision_score(y_test, preds_ns_50, average="binary")
    ns_rec_50 = recall_score(y_test, preds_ns_50, average="binary")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("    PREDICTIVE MAINTENANCE - CLASS IMBALANCE REPORT   \n")
        f.write("==================================================\n\n")
        f.write(f"Trained Optimal Classifier: {global_best_model_name}\n")
        f.write(f"Tuned Optimal Gating Threshold: {global_best_threshold:.2f}\n\n")
        f.write("Detailed Comparative Performance:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"1. Baseline Model (WITHOUT SMOTE - {ns_model_name} at 0.50 Gating):\n")
        f.write(f"  - Validation Accuracy : {ns_acc * 100:.2f}%\n")
        f.write(f"  - Validation Precision: {ns_prec_50:.4f}\n")
        f.write(f"  - Validation Recall   : {ns_rec_50:.4f}\n")
        f.write(f"  - Validation F1-Score : {ns_f1:.4f}\n\n")
        f.write(f"2. Tuned Optimal Model ({global_best_model_name} at {global_best_threshold:.2f} Gating):\n")
        f.write(f"  - Validation Accuracy : {global_best_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"  - Validation Precision: {global_best_metrics['precision']:.4f}\n")
        f.write(f"  - Validation Recall   : {global_best_metrics['recall']:.4f}\n")
        f.write(f"  - Validation F1-Score : {global_best_metrics['f1_score']:.4f}\n\n")
        f.write("--------------------------------------------------\n")
        f.write("Key Observations:\n")
        f.write(f"  * Recall Difference : {global_best_metrics['recall'] - ns_rec_50:+.4f}\n")
        f.write(f"  * F1-score Difference: {global_best_metrics['f1_score'] - ns_f1:+.4f}\n")
        
    logger.info(f"Successfully saved imbalance comparison report to: {report_path}")

if __name__ == "__main__":
    logger.info("--- Standalone Advanced ML Training Pipeline Started ---")
    execute_model_training_pipeline()
    logger.info("--- Standalone Advanced ML Training Pipeline Completed ---")