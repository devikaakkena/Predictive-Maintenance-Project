import pandas as pd
from flask import Blueprint, render_template, request
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService

prediction_bp = Blueprint("prediction", __name__)
prediction_service = PredictionService()

@prediction_bp.route("/predict", methods=["POST"])
def predict():
    # Extract values from request
    air_temp = float(request.form["air_temp"])
    process_temp = float(request.form["process_temp"])
    speed = float(request.form["speed"])
    torque = float(request.form["torque"])
    tool_wear = float(request.form["tool_wear"])
    
    # Predict single instance
    features = [air_temp, process_temp, speed, torque, tool_wear]
    result = prediction_service.predict_single(features)
    
    # Reload and predict batch to keep dataset table updated
    data = pd.read_csv(Config.DATA_PATH)
    predictions_df = prediction_service.predict_batch(data)
    safe_count, fail_count = AnalysisService.get_prediction_counts(predictions_df)
    
    return render_template(
        "index.html",
        prediction_text=result,
        table=predictions_df.to_html(index=False),
        safe=safe_count,
        failure=fail_count
    )
