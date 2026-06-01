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

    def get_recent_predictions(self) -> list:
        """Loads the latest 10 predictions from the SQLite database, with JSON backup fallback."""
        try:
            from backend.app.services.database_service import DatabaseService
            db_predictions = DatabaseService.get_recent_predictions(limit=10)
            if db_predictions:
                return db_predictions
        except Exception as e:
            predictions_logger.warning(f"Database query failed, falling back to JSON backup: {str(e)}")
            
        from pathlib import Path
        import json
        base_dir = Path(__file__).resolve().parents[4]
        json_path = base_dir / "outputs" / "predictions" / "recent_predictions.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def add_recent_prediction(self, prediction_entry: dict) -> None:
        """Saves a new prediction to outputs/predictions/recent_predictions.json and truncates to 10 entries."""
        from pathlib import Path
        import json
        base_dir = Path(__file__).resolve().parents[4]
        json_dir = base_dir / "outputs" / "predictions"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / "recent_predictions.json"
        
        recent = self.get_recent_predictions()
        recent.insert(0, prediction_entry)
        recent = recent[:10]  # Keep latest 10 only
        
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(recent, f, indent=4)
        except Exception as e:
            errors_logger.error(f"Failed to save recent prediction to JSON: {str(e)}")

    def predict_single_detailed(self, features: list) -> dict:
        """Runs detailed prediction on a single record and returns rich classification indicators."""
        import time
        try:
            # Scale features before prediction
            scaled_features = self.scaler.transform([features])[0]
            
            prob = 0.0
            if hasattr(self.model, "predict_proba"):
                prob = float(self.model.predict_proba([scaled_features])[0][1])
                prediction = 1 if prob >= self.threshold else 0
            else:
                prediction = int(self.model.predict([scaled_features])[0])
                prob = 1.0 if prediction == 1 else 0.0
                
            # Confidence is the probability of the chosen class
            confidence = prob if prediction == 1 else (1.0 - prob)
            
            # Determine status badge
            air_temp, process_temp, speed, torque, tool_wear = features
            if prediction == 1:
                status = "CRITICAL"
                status_color = "danger"
            elif tool_wear > 175 or torque > 50 or prob >= 0.40:
                status = "WARNING"
                status_color = "warning"
            else:
                status = "SAFE"
                status_color = "success"
                
            label = format_prediction(prediction)
            
            prediction_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sensor_summary": f"Air: {air_temp:.1f}K | Speed: {speed:.0f}rpm | Torque: {torque:.1f}Nm",
                "result": label,
                "confidence": int(confidence * 100),
                "status": status,
                "status_color": status_color,
                "raw_features": features
            }
            
            # Save persistently to SQLite database
            try:
                from backend.app.services.database_service import DatabaseService
                DatabaseService.save_prediction(
                    features=features,
                    prediction=label,
                    confidence_score=float(confidence * 100),
                    machine_status=status
                )
            except Exception as db_err:
                predictions_logger.error(f"Failed to persist prediction in SQLite database: {str(db_err)}")
                
            self.add_recent_prediction(prediction_entry)
            
            predictions_logger.info(f"Detailed Inference: {prediction_entry}")
            return prediction_entry
        except Exception as e:
            errors_logger.error(f"Failed to run detailed model inference: {str(e)}")
            raise e
