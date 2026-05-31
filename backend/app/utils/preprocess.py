import pandas as pd
from backend.app.config.settings import Config

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts only the required feature columns for the ML model."""
    return df[Config.FEATURE_COLUMNS]
