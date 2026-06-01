import os
from pathlib import Path
from flask import Blueprint, render_template, send_from_directory
from backend.app.services.dashboard_service import DashboardService
from backend.app.utils.logger import app_logger, errors_logger

analysis_bp = Blueprint("analysis", __name__)
dashboard_service = DashboardService()

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

@analysis_bp.route("/analysis")
def index():
    app_logger.info("Serving dynamic analysis dashboard page...")
    report_content = "Run the ML evaluation pipeline to generate statistical reports."
    
    # Scan for the latest classification report dynamically
    if REPORTS_DIR.exists():
        report_files = sorted(
            [f for f in REPORTS_DIR.glob("classification_report_*.txt")],
            key=os.path.getmtime,
            reverse=True
        )
        if report_files:
            try:
                with open(report_files[0], "r", encoding="utf-8") as f:
                    report_content = f.read()
                app_logger.info(f"Loaded classification report: {report_files[0].name}")
            except Exception as e:
                report_content = f"Error reading validation report: {str(e)}"
                errors_logger.error(f"Failed to read validation report file: {str(e)}")
                
    try:
        # Fetch optimal model metrics dynamically
        metrics = dashboard_service.get_model_performance_metrics()
        
        # Load dynamic insights at runtime
        import json
        insights = {}
        insights_path = BASE_DIR / "ml" / "models" / "pipeline_insights.json"
        if insights_path.exists():
            try:
                with open(insights_path, "r", encoding="utf-8") as f:
                    insights = json.load(f)
            except Exception as e:
                errors_logger.error(f"Failed to read pipeline_insights.json: {str(e)}")
        
        return render_template(
            "dashboard/analytics.html",
            report_text=report_content,
            precision=f"{metrics['precision']:.4f}",
            recall=f"{metrics['recall']:.4f}",
            f1_score=f"{metrics['f1_score']:.4f}",
            insights=insights
        )
    except Exception as e:
        errors_logger.error(f"Error rendering analysis page: {str(e)}")
        return "<h3>❌ Failed to load analysis page. Check logs for details.</h3>", 500

@analysis_bp.route("/outputs/graphs/<path:filename>")
def serve_graph(filename):
    """Serves generated Matplotlib/Seaborn graph images from the outputs directory."""
    graphs_dir = BASE_DIR / "outputs" / "graphs"
    return send_from_directory(graphs_dir, filename)

