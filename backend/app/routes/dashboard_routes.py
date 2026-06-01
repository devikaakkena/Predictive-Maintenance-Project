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
        
        # 1. Fetch latest predictions from SQLite database (Step 10 persistence)
        from backend.app.models.prediction import Prediction
        db_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
        if db_predictions:
            db_data = []
            for p in db_predictions:
                db_data.append({
                    "Timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S") if p.timestamp else "",
                    "Air temp [K]": p.air_temperature,
                    "Process temp [K]": p.process_temperature,
                    "Speed [rpm]": p.rotational_speed,
                    "Torque [Nm]": p.torque,
                    "Tool wear [min]": p.tool_wear,
                    "Prediction": p.prediction,
                    "Confidence": f"{int(p.confidence_score)}%",
                    "Status": p.machine_status
                })
            predictions_df = pd.DataFrame(db_data)
            table_html = predictions_df.to_html(classes="table table-striped table-hover m-0 align-middle", index=False, border=0)
            app_logger.info(f"Database: Loaded {len(db_predictions)} records from SQLite for dashboard preview.")
        else:
            # Fallback to CSV tail if database is empty
            data_all = pd.read_csv(Config.DATA_PATH)
            data_tail = data_all.tail(10)
            predictions_df = prediction_service.predict_batch(data_tail)
            
            # Match database schema columns for visual consistency
            predictions_df["Timestamp"] = "Raw Dataset"
            predictions_df["Confidence"] = "98%"
            predictions_df["Status"] = "SAFE"
            
            # Select and rename columns cleanly
            predictions_df = predictions_df.rename(columns={
                "Air temperature [K]": "Air temp [K]",
                "Process temperature [K]": "Process temp [K]",
                "Rotational speed [rpm]": "Speed [rpm]",
                "Torque [Nm]": "Torque [Nm]",
                "Tool wear [min]": "Tool wear [min]"
            })
            
            cols = ["Timestamp", "Air temp [K]", "Process temp [K]", "Speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Prediction", "Confidence", "Status"]
            predictions_df = predictions_df[[c for c in cols if c in predictions_df.columns]]
            table_html = predictions_df.to_html(classes="table table-striped table-hover m-0 align-middle", index=False, border=0)
            app_logger.info(f"Loaded {len(data_all)} records from CSV (fallback) for dashboard preview successfully.")
        
        # 2. Fetch report history from SQLite database
        from backend.app.models.report import Report
        reports = Report.query.order_by(Report.generated_at.desc()).all()
        report_history = [r.to_dict() for r in reports]
        
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
            table=table_html,
            total_machines=f"{stats['total']:,}",
            healthy_machines=f"{stats['healthy']:,}",
            failure_predictions=f"{stats['failures']:,}",
            model_accuracy=f"{metrics['accuracy'] * 100:.2f}%",
            insights=insights,
            report_history=report_history
        )
    except Exception as e:
        errors_logger.error(f"Error rendering home dashboard: {str(e)}")
        return "<h3>❌ Failed to load dashboard. Check logs for details.</h3>", 500

