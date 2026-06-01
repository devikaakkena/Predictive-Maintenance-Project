import os
import joblib
from backend.app.config.settings import Config

def test_model_path_exists():
    """Validates that the pre-trained model file exists at the configured path."""
    assert os.path.exists(Config.MODEL_PATH), f"Model file not found at {Config.MODEL_PATH}"

def test_model_loading_and_inference():
    """Validates that the serialized model can be successfully loaded and run inference on mock data."""
    assert os.path.exists(Config.MODEL_PATH)
    model = joblib.load(Config.MODEL_PATH)
    assert model is not None
    
    # Air Temp [K], Process Temp [K], Speed [rpm], Torque [Nm], Tool Wear [min]
    mock_features = [300.0, 310.0, 1500.0, 40.0, 50.0]
    prediction = model.predict([mock_features])[0]
    assert prediction in [0, 1], "Prediction output must be binary (0 for healthy, 1 for failure)"
