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
        
        # Load gating threshold dynamically from threshold_results.json
        from pathlib import Path
        import json
        self.threshold = 0.50
        base_dir = Path(__file__).resolve().parents[4]
        threshold_path = base_dir / "ml" / "models" / "threshold_results.json"
        if threshold_path.exists():
            try:
                with open(threshold_path, "r", encoding="utf-8") as f:
                    th_data = json.load(f)
                    self.threshold = th_data.get("optimal_threshold", 0.50)
                predictions_logger.info(f"Loaded optimal gating threshold dynamically: {self.threshold:.2f}")
            except Exception as e:
                predictions_logger.warning(f"Failed to read threshold_results.json: {str(e)}. Defaulting to 0.50.")

    def predict_single(self, features: list) -> str:
        """Runs prediction on a single record and returns the formatted label."""
        try:
            # Scale features before prediction
            scaled_features = self.scaler.transform([features])[0]
            
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba([scaled_features])[0][1]
                prediction = 1 if prob >= self.threshold else 0
            else:
                prediction = self.model.predict([scaled_features])[0]
                
            label = format_prediction(prediction)
            predictions_logger.info(f"Model Inference: Features {features} (scaled: {scaled_features.tolist()}) -> Prediction: {label} (threshold: {self.threshold:.2f})")
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
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_scaled)[:, 1]
                predictions = (probs >= self.threshold).astype(int)
            else:
                predictions = self.model.predict(X_scaled)
            
            df_copy = df.copy()
            df_copy["Prediction"] = [format_prediction(p) for p in predictions]
            
            failures_count = sum(1 for p in predictions if p != 0)
            predictions_logger.info(
                f"Model Inference: Completed batch prediction. Flagged {failures_count} failure conditions out of {len(df)} records (threshold: {self.threshold:.2f})."
            )
            return df_copy
        except Exception as e:
            errors_logger.error(f"Failed to run model batch inference: {str(e)}")
            raise e


