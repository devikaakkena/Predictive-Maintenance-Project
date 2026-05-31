import pandas as pd
from flask import Blueprint, render_template
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService

from backend.app.services.dashboard_service import DashboardService

dashboard_bp = Blueprint("dashboard", __name__)
prediction_service = PredictionService()
dashboard_service = DashboardService()

@dashboard_bp.route("/")
def home():
    # Fetch real stats and metrics dynamically
    stats = dashboard_service.get_machine_stats()
    metrics = dashboard_service.get_model_performance_metrics()
    
    # Load dataset for predictions logs table
    data = pd.read_csv(Config.DATA_PATH)
    predictions_df = prediction_service.predict_batch(data)
    
    return render_template(
        "dashboard/index.html",
        table=predictions_df.to_html(classes="table table-striped table-hover m-0 align-middle", index=False, border=0),
        total_machines=f"{stats['total']:,}",
        healthy_machines=f"{stats['healthy']:,}",
        failure_predictions=f"{stats['failures']:,}",
        model_accuracy=f"{metrics['accuracy'] * 100:.2f}%"
    )
