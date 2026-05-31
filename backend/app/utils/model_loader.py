import joblib
from backend.app.config.settings import Config

def load_prediction_model():
    """Loads the pre-trained machine learning model from disk."""
    return joblib.load(Config.MODEL_PATH)
