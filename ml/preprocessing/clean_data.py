import pandas as pd
import logging

# Configure local module logger
logger = logging.getLogger(__name__)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input dataset by removing duplicate rows and handling missing values.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logger.info("Starting dataset cleaning process...")
    initial_shape = df.shape
    
    # 1. Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} missing values in dataset. Dropping null records...")
        df = df.dropna()
    else:
        logger.info("Check complete: No missing values found.")
        
    # 2. Handle duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate rows. Dropping duplicates...")
        df = df.drop_duplicates()
    else:
        logger.info("Check complete: No duplicate rows found.")
        
    final_shape = df.shape
    logger.info(f"Dataset cleaning completed. Initial shape: {initial_shape}, Cleaned shape: {final_shape}")
    return df
