from flask import Blueprint, jsonify
from backend.app.services.dashboard_service import DashboardService
from backend.app.utils.logger import app_logger, errors_logger

# Define cleanly separated API Blueprint with a dedicated url prefix
api_bp = Blueprint("api", __name__, url_prefix="/api")
dashboard_service = DashboardService()

@api_bp.route("/dashboard-metrics", methods=["GET"])
def get_dashboard_metrics():
    """Returns total counts and best classifier validation scores."""
    app_logger.info("API: GET /api/dashboard-metrics request received.")
    try:
        stats = dashboard_service.get_machine_stats()
        metrics = dashboard_service.get_model_performance_metrics()
        return jsonify({
            "total": stats["total"],
            "healthy": stats["healthy"],
            "failures": stats["failures"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })
    except Exception as e:
        errors_logger.error(f"API Error in /dashboard-metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/model-comparison", methods=["GET"])
def get_model_comparison():
    """Returns comparison metrics for all trained models alongside confusion matrix details."""
    app_logger.info("API: GET /api/model-comparison request received.")
    try:
        metrics = dashboard_service.get_all_models_comparison_metrics()
        cm = dashboard_service.get_confusion_matrix()
        return jsonify({
            "metrics": metrics,
            "confusion_matrix": cm
        })
    except Exception as e:
        errors_logger.error(f"API Error in /model-comparison: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/prediction-summary", methods=["GET"])
def get_prediction_summary():
    """Returns binary class distribution parameters for operations."""
    app_logger.info("API: GET /api/prediction-summary request received.")
    try:
        stats = dashboard_service.get_machine_stats()
        return jsonify({
            "labels": ["Healthy Operations", "Flagged Failures"],
            "values": [stats["healthy"], stats["failures"]]
        })
    except Exception as e:
        errors_logger.error(f"API Error in /prediction-summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """Returns feature importances for optimal model."""
    app_logger.info("API: GET /api/feature-importance request received.")
    try:
        importances_data = dashboard_service.get_feature_importances()
        return jsonify(importances_data)
    except Exception as e:
        errors_logger.error(f"API Error in /feature-importance: {str(e)}")
        return jsonify({"error": str(e)}), 500

