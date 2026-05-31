import os

class Config:
    # Resolve the absolute workspace root path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    # Path configuration compatible with Windows
    DATA_PATH = os.path.join(BASE_DIR, "ml", "data", "raw", "ai4i.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "trained_model.pkl")
    
    FEATURE_COLUMNS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    SECRET_KEY = os.environ.get("SECRET_KEY", "predictive_maintenance_secret_key")
