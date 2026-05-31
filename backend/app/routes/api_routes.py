from flask import Blueprint, jsonify
from backend.app.services.dashboard_service import DashboardService

# Define cleanly separated API Blueprint with a dedicated url prefix
api_bp = Blueprint("api", __name__, url_prefix="/api")
dashboard_service = DashboardService()

@api_bp.route("/dashboard-metrics", methods=["GET"])
def get_dashboard_metrics():
    """Returns total counts and best classifier validation scores."""
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

@api_bp.route("/model-comparison", methods=["GET"])
def get_model_comparison():
    """Returns comparison metrics for all trained models alongside confusion matrix details."""
    metrics = dashboard_service.get_all_models_comparison_metrics()
    cm = dashboard_service.get_confusion_matrix()
    return jsonify({
        "metrics": metrics,
        "confusion_matrix": cm
    })

@api_bp.route("/prediction-summary", methods=["GET"])
def get_prediction_summary():
    """Returns binary class distribution parameters for operations."""
    stats = dashboard_service.get_machine_stats()
    return jsonify({
        "labels": ["Healthy Operations", "Flagged Failures"],
        "values": [stats["healthy"], stats["failures"]]
    })

@api_bp.route("/feature-importance", methods=["GET"])
def get_feature_importance():
    """Returns feature importances for optimal model."""
    importances_data = dashboard_service.get_feature_importances()
    return jsonify(importances_data)
