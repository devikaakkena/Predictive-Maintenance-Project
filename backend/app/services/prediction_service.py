import pandas as pd
from backend.app.utils.model_loader import load_prediction_model
from backend.app.utils.preprocess import extract_features
from backend.app.utils.helpers import format_prediction

class PredictionService:
    def __init__(self):
        # Cache the loaded model
        self.model = load_prediction_model()

    def predict_single(self, features: list) -> str:
        """Runs prediction on a single record and returns the formatted label."""
        prediction = self.model.predict([features])[0]
        return format_prediction(prediction)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends a 'Prediction' column to the dataframe with formatted labels."""
        X = extract_features(df)
        predictions = self.model.predict(X)
        
        df_copy = df.copy()
        df_copy["Prediction"] = [format_prediction(p) for p in predictions]
        return df_copy
