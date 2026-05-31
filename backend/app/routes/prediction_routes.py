import pandas as pd
from flask import Blueprint, render_template, request
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService

prediction_bp = Blueprint("prediction", __name__)
prediction_service = PredictionService()

@prediction_bp.route("/predictions", methods=["GET"])
def predict_view():
    # Render the input parameters form page
    return render_template("dashboard/predictions.html")

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
    
    return render_template(
        "dashboard/predictions.html",
        prediction_text=result
    )
