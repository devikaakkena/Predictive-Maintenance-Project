import joblib
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Import cleanly using absolute paths from modular structures
from ml.preprocessing.clean_data import clean_dataset
from ml.preprocessing.feature_engineering import select_features_and_labels

# Configure system logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute directory paths safely via pathlib
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "ml" / "data" / "raw" / "ai4i.csv"
SCALER_SAVE_PATH = BASE_DIR / "ml" / "models" / "scaler.pkl"

def ensure_output_directories() -> None:
    """Ensures all target outputs subdirectories exist on disk dynamically."""
    outputs_base = BASE_DIR / "outputs"
    subdirs = ["graphs", "reports", "predictions", "logs"]
    for sd in subdirs:
        dir_path = outputs_base / sd
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified directory existence: {dir_path}")

def run_preprocessing_pipeline(
    data_path: Path = RAW_DATA_PATH,
    scaler_save_path: Path = SCALER_SAVE_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Tuple]:
    """
    Orchestrates the complete data preprocessing pipeline.
    
    1. Ensures target outputs folders exist
    2. Loads raw dataset from raw directory
    3. Cleans dataset (handles missing values and duplicate rows)
    4. Extracts features and target labels separately
    5. Splits dataset into train/test sets, guarding against data leakage
    6. Fits StandardScaler on training set only, then scales both splits
    7. Persists fitted StandardScaler as scaler.pkl for inference use
    8. Saves intermediate processed datasets to processed/ directory
    
    Args:
        data_path (Path): Path to raw input csv file.
        scaler_save_path (Path): Path where to save the scaler.pkl.
        test_size (float): Proportion of dataset in the test split.
        random_state (int): Configurable seed for reproducibility.
        
    Returns:
        Dict[str, Tuple]: Dictionary holding scaled (X, y) splits for 'train' and 'test'.
    """
    logger.info("Executing Predictive Maintenance Preprocessing Pipeline...")
    
    # 1. Ensure target outputs subdirectories exist
    ensure_output_directories()
    
    # 2. Load dataset
    if not data_path.exists():
        logger.error(f"Dataset not found at absolute path: {data_path}")
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw dataset from: {data_path} (Shape: {df.shape})")
    
    # 2. Clean dataset
    df_cleaned = clean_dataset(df)
    
    # 3. Feature Selection
    X, y = select_features_and_labels(df_cleaned)
    
    # 4. Train-Test Split (stratified on label to preserve class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train-Test Split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Fit & Transform StandardScaler (fit ONLY on training set to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Successfully fitted StandardScaler on training split and scaled both splits.")
    
    # 6. Save scaler.pkl weight
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"Persisted scaler model weight successfully to: {scaler_save_path}")
    
    # 7. Save intermediate processed splits to ml/data/processed/
    processed_dir = BASE_DIR / "ml" / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSVs for subsequent training phases
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(processed_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    logger.info(f"Saved intermediate processed training splits to: {processed_dir}")
    
    return {
        "train": (X_train_scaled, y_train),
        "test": (X_test_scaled, y_test)
    }

if __name__ == "__main__":
    logger.info("--- Standalone Preprocessing Pipeline Execution Started ---")
    run_preprocessing_pipeline()
    logger.info("--- Standalone Preprocessing Pipeline Completed Successfully ---")
