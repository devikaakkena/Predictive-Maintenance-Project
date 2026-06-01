import pandas as pd
from flask import Blueprint, render_template, request
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService
from backend.app.utils.logger import predictions_logger, errors_logger

prediction_bp = Blueprint("prediction", __name__)
prediction_service = PredictionService()

@prediction_bp.route("/predictions", methods=["GET"])
def predict_view():
    predictions_logger.info("Serving predictions manual inference form page.")
    recent_predictions = prediction_service.get_recent_predictions()
    # Render the input parameters form page
    return render_template(
        "dashboard/predictions.html",
        recent_predictions=recent_predictions
    )

@prediction_bp.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract values from request
        air_temp = float(request.form["air_temp"])
        process_temp = float(request.form["process_temp"])
        speed = float(request.form["speed"])
        torque = float(request.form["torque"])
        tool_wear = float(request.form["tool_wear"])
        
        predictions_logger.info(
            f"Manual Inference Request - AirTemp: {air_temp}K, ProcessTemp: {process_temp}K, "
            f"Speed: {speed}rpm, Torque: {torque}Nm, ToolWear: {tool_wear}min"
        )
        
        # Predict single instance with detailed outputs
        features = [air_temp, process_temp, speed, torque, tool_wear]
        prediction_detail = prediction_service.predict_single_detailed(features)
        recent_predictions = prediction_service.get_recent_predictions()
        
        return render_template(
            "dashboard/predictions.html",
            prediction_text=prediction_detail["result"],
            prediction_detail=prediction_detail,
            recent_predictions=recent_predictions
        )
    except Exception as e:
        errors_logger.error(f"Failed to run manual inference: {str(e)}")
        recent_predictions = prediction_service.get_recent_predictions()
        return render_template(
            "dashboard/predictions.html",
            prediction_text=f"Error running prediction: {str(e)}",
            recent_predictions=recent_predictions
        )

