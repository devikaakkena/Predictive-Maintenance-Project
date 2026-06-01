import pandas as pd
from backend.app.utils.model_loader import load_prediction_model, load_scaler
from backend.app.utils.preprocess import extract_features
from backend.app.utils.helpers import format_prediction
from backend.app.utils.logger import predictions_logger, errors_logger

class PredictionService:
    def __init__(self):
        # Cache the loaded model and scaler
        self.model = load_prediction_model()
        self.scaler = load_scaler()

    def predict_single(self, features: list) -> str:
        """Runs prediction on a single record and returns the formatted label."""
        try:
            # Scale features before prediction
            scaled_features = self.scaler.transform([features])[0]
            prediction = self.model.predict([scaled_features])[0]
            label = format_prediction(prediction)
            predictions_logger.info(f"Model Inference: Features {features} (scaled: {scaled_features.tolist()}) -> Prediction: {label}")
            return label
        except Exception as e:
            errors_logger.error(f"Failed to run model inference on features {features}: {str(e)}")
            raise e

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appends a 'Prediction' column to the dataframe with formatted labels."""
        try:
            predictions_logger.info(f"Model Inference: Running batch prediction on {len(df)} records...")
            X = extract_features(df)
            # Scale features before prediction
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            df_copy = df.copy()
            df_copy["Prediction"] = [format_prediction(p) for p in predictions]
            
            failures_count = sum(1 for p in predictions if p != 0)
            predictions_logger.info(
                f"Model Inference: Completed batch prediction. Flagged {failures_count} failure conditions out of {len(df)} records."
            )
            return df_copy
        except Exception as e:
            errors_logger.error(f"Failed to run model batch inference: {str(e)}")
            raise e


