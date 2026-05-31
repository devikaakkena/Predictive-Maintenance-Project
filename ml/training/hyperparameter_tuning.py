import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Configure local module logger
logger = logging.getLogger(__name__)

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Performs grid search hyperparameter tuning for Random Forest.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        
    Returns:
        RandomForestClassifier: Tuned estimator fitted on the dataset.
    """
    logger.info("Initializing hyperparameter optimization grid search for Random Forest...")
    
    # Grid parameters
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Hyperparameter optimization completed. Selected Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
