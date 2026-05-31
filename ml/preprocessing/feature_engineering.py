import pandas as pd
import logging
from typing import Tuple

# Configure local module logger
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

TARGET_COLUMN = 'Machine failure'

def select_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the feature columns (X) and target label column (y) from the cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): Cleaned input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix (X) and target labels (y).
    """
    logger.info("Extracting feature columns and target label column...")
    
    # Check if target column is present
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        
    # Check if all required features are present
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        logger.error(f"Missing required predictive features: {missing_features}")
        raise ValueError(f"Required features {missing_features} are missing from the dataset.")
        
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    logger.info(f"Feature selection complete. Matrix shape: {X.shape}, Label shape: {y.shape}")
    return X, y
