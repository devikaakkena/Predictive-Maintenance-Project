import pandas as pd
from flask import Blueprint, render_template
from backend.app.config.settings import Config
from backend.app.services.prediction_service import PredictionService
from backend.app.services.analysis_service import AnalysisService
from backend.app.services.dashboard_service import DashboardService
from backend.app.utils.logger import app_logger, errors_logger

dashboard_bp = Blueprint("dashboard", __name__)
prediction_service = PredictionService()
dashboard_service = DashboardService()

@dashboard_bp.route("/")
def home():
    app_logger.info("Serving dynamic predictive maintenance dashboard page...")
    try:
        # Fetch real stats and metrics dynamically
        stats = dashboard_service.get_machine_stats()
        metrics = dashboard_service.get_model_performance_metrics()
        
        # Load last 10 records for predictions logs preview table to avoid bloated page loading times
        data_all = pd.read_csv(Config.DATA_PATH)
        data_tail = data_all.tail(10)
        predictions_df = prediction_service.predict_batch(data_tail)
        
        app_logger.info(f"Loaded {len(data_all)} records (displaying latest 10) for dashboard preview successfully.")
        
        # Load dynamic insights at runtime
        from pathlib import Path
        import json
        insights = {}
        base_dir = Path(__file__).resolve().parents[3]
        insights_path = base_dir / "ml" / "models" / "pipeline_insights.json"
        if insights_path.exists():
            try:
                with open(insights_path, "r", encoding="utf-8") as f:
                    insights = json.load(f)
            except Exception as e:
                errors_logger.error(f"Failed to read pipeline_insights.json: {str(e)}")
                
        return render_template(
            "dashboard/index.html",
            table=predictions_df.to_html(classes="table table-striped table-hover m-0 align-middle", index=False, border=0),
            total_machines=f"{stats['total']:,}",
            healthy_machines=f"{stats['healthy']:,}",
            failure_predictions=f"{stats['failures']:,}",
            model_accuracy=f"{metrics['accuracy'] * 100:.2f}%",
            insights=insights
        )
    except Exception as e:
        errors_logger.error(f"Error rendering home dashboard: {str(e)}")
        return "<h3>❌ Failed to load dashboard. Check logs for details.</h3>", 500

