import os
from dotenv import load_dotenv

# Resolve the absolute workspace root path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Load environment variables from the .env file at the project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Config:
    """
    Centralized configuration manager loading environment variable overrides 
    with robust local fallbacks.
    """
    BASE_DIR = BASE_DIR
    
    # Primary telemetry raw dataset path
    DATA_PATH = os.path.join(BASE_DIR, "ml", "data", "raw", "ai4i.csv")
    
    # Environment Configurations with default fallbacks
    SECRET_KEY = os.environ.get("SECRET_KEY", "predictive_maintenance_secret_key")
    DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")
    
    # Machine Learning optimal model files paths overrides
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "ml", "models", "trained_model.pkl")
    )
    SCALER_PATH = os.environ.get(
        "SCALER_PATH",
        os.path.join(BASE_DIR, "ml", "models", "scaler.pkl")
    )
    
    # SQLite Database URI configuration overrides with absolute path auto-resolution
    raw_db_uri = os.environ.get("DATABASE_URI", "")
    if raw_db_uri:
        if raw_db_uri.startswith("sqlite:///instance/"):
            # Ensure absolute instance path resolution to handle all Windows workspace run levels
            db_filename = raw_db_uri.replace("sqlite:///instance/", "")
            SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'instance', db_filename)}"
        else:
            SQLALCHEMY_DATABASE_URI = raw_db_uri
    else:
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'predictive_maintenance.db')}"
        
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Dynamic PDF Serving Output Folder Path Configuration
    raw_report_path = os.environ.get("REPORT_OUTPUT_PATH", "")
    if raw_report_path:
        if os.path.isabs(raw_report_path):
            REPORT_OUTPUT_PATH = raw_report_path
        else:
            REPORT_OUTPUT_PATH = os.path.abspath(os.path.join(BASE_DIR, raw_report_path))
    else:
        REPORT_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "reports")
    
    # Dynamic logs level overrides (INFO, DEBUG, WARNING, ERROR)
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    
    # UCI Standard feature columns
    FEATURE_COLUMNS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
