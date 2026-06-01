import joblib
from backend.app.config.settings import Config
from backend.app.utils.logger import app_logger, errors_logger

def load_prediction_model():
    """Loads the pre-trained machine learning model from disk."""
    try:
        app_logger.info(f"Loading pre-trained machine learning model from: {Config.MODEL_PATH}")
        model = joblib.load(Config.MODEL_PATH)
        app_logger.info("Successfully loaded pre-trained machine learning model.")
        return model
    except Exception as e:
        errors_logger.error(f"Failed to load prediction model from {Config.MODEL_PATH}: {str(e)}")
        raise e

