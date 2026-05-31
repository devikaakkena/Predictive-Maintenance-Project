import pandas as pd
import logging
import shutil
from pathlib import Path

# Absolute imports from training packages
from ml.training.compare_models import compare_multiple_models, select_best_model
from ml.training.hyperparameter_tuning import tune_random_forest
from ml.training.save_model import save_best_model_and_metadata

# Configure logging module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "ml" / "data" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"

def execute_model_training_pipeline() -> None:
    """
    Executes the entire modular machine learning model training pipeline.
    
    1. Loads intermediate preprocessed splits from processed directory
    2. Trains and compares Logistic Regression, Decision Tree, Random Forest, and XGBoost
    3. Automatically selects the optimal classifier based on F1 validation performance
    4. Automatically runs hyperparameter tuning on Random Forest if it is the best
    5. Saves best model weights, JSON performance profiles, and a comparisons report
    6. Ensures backend Flask app compatibility by linking/copying best_model.pkl to trained_model.pkl
    """
    logger.info("Executing Predictive Maintenance Model Training Pipeline...")
    
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
    
    # 3. Train and compare models
    fitted_models, metrics = compare_multiple_models(X_train, y_train, X_test, y_test)
    
    # 4. Select best model
    best_name = select_best_model(metrics)
    best_model = fitted_models[best_name]
    
    # 5. Grid Search Tuning if Random Forest is selected
    if best_name == "Random Forest":
        tuned_model = tune_random_forest(X_train, y_train)
        
        # Fit and update metrics
        tuned_model.fit(X_train, y_train)
        y_pred = tuned_model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        
        metrics["Random Forest (Tuned)"] = {
            "accuracy": float(acc),
            "f1_score": float(f1)
        }
        best_model = tuned_model
        best_name = "Random Forest (Tuned)"
        logger.info(f"Tuned Random Forest Validation Accuracy: {acc*100:.2f}%, F1-Score: {f1:.4f}")
        
    # 6. Save model weight pickles and metrics metadata
    save_best_model_and_metadata(best_model, best_name, metrics)
    
    # 7. Maintain Backend Flask application compatibility
    trained_model_path = MODELS_DIR / "trained_model.pkl"
    shutil.copy(MODELS_DIR / "best_model.pkl", trained_model_path)
    logger.info(f"Successfully cloned best_model.pkl to {trained_model_path} for backward backend compatibility.")

if __name__ == "__main__":
    logger.info("--- Standalone Model Training Pipeline Started ---")
    execute_model_training_pipeline()
    logger.info("--- Standalone Model Training Pipeline Completed Successfully ---")