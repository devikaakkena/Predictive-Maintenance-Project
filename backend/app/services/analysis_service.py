import pandas as pd

class AnalysisService:
    @staticmethod
    def get_prediction_counts(df: pd.DataFrame) -> tuple:
        """Returns the number of safe and failed machines in the dataset."""
        if "Prediction" not in df.columns:
            return 0, 0
        safe_count = int((df["Prediction"] == "✅ Safe").sum())
        fail_count = int((df["Prediction"] == "⚠️ Failure").sum())
        return safe_count, fail_count
