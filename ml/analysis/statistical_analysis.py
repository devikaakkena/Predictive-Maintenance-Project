import pandas as pd
import logging

# Configure logger
logger = logging.getLogger(__name__)

def perform_dataset_statistical_analysis(df: pd.DataFrame) -> None:
    """
    Computes and logs descriptive statistics of the variables in the dataset.
    
    Args:
        df (pd.DataFrame): Data matrix to analyze.
    """
    logger.info("Computing descriptive statistics summary for test features split...")
    stats_summary = df.describe().to_string()
    
    logger.info(f"\n--- Descriptive Statistics Summary ---\n{stats_summary}\n")
