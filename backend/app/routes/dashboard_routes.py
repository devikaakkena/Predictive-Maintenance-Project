import pandas as pd
from flask import Blueprint, render_template
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService

dashboard_bp = Blueprint("dashboard", __name__)
prediction_service = PredictionService()

@dashboard_bp.route("/")
def home():
    # Load dataset
    data = pd.read_csv(Config.DATA_PATH)
    
    # Predict batch
    predictions_df = prediction_service.predict_batch(data)
    
    # Get summary counts
    safe_count, fail_count = AnalysisService.get_prediction_counts(predictions_df)
    
    return render_template(
        "index.html",
        table=predictions_df.to_html(index=False),
        safe=safe_count,
        failure=fail_count
    )
