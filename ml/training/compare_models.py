import logging
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set local module logger
logger = logging.getLogger(__name__)

# Try importing XGBoost defensively with fallback compatibility
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info("XGBoost library is available and loaded.")
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost library is unavailable. Comparative fallback models will be trained.")

def compare_multiple_models(
    X_train, y_train, X_test, y_test
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Trains multiple machine learning models and compares accuracy and F1 scores.
    
    Required models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. XGBoost (optional fallback if package is missing)
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
            - Dictionary of trained model instances.
            - Dictionary of model metrics.
    """
    logger.info("Beginning multiple model training comparisons...")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Fallback check
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=42, n_estimators=100, eval_metric="logloss"
        )
        
    fitted_models = {}
    metrics = {}
    
    # Train each model and calculate standard validation performance
    for name, model in models.items():
        logger.info(f"Training classification model: {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Run inference on test split
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        
        metrics[name] = {
            "accuracy": float(acc),
            "f1_score": float(f1)
        }
        logger.info(f"{name} results -> Accuracy: {acc * 100:.2f}%, F1-Score: {f1:.4f}")
        
    return fitted_models, metrics

def select_best_model(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Automatically selects the best model based on F1-score performance.
    
    Args:
        metrics (Dict): Dictionary containing compiled model metrics.
        
    Returns:
        str: Name of the optimal model.
    """
    best_model_name = None
    best_f1_score = -1.0
    
    for name, scores in metrics.items():
        # F1-score is preferred for predictive maintenance (due to class imbalance)
        f1 = scores["f1_score"]
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            
    logger.info(f"Optimal model selection: '{best_model_name}' has the highest F1-Score of {best_f1_score:.4f}")
    return best_model_name
