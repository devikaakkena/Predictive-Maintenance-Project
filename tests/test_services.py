import os
import pandas as pd
from backend.app.services.prediction_service import PredictionService
from backend.app.services.dashboard_service import DashboardService
from backend.app.services.simulation_service import SimulationService
from backend.app.services.report_service import ReportService
from backend.app.config.settings import Config

def test_prediction_service():
    """Validates single record and batch prediction output formats inside PredictionService."""
    service = PredictionService()
    
    # 1. Test single inference
    mock_features = [300.0, 310.0, 1500.0, 40.0, 50.0]
    result = service.predict_single(mock_features)
    assert result in ["✅ Safe", "⚠️ Failure"], "Outcome must map to formatted string flags."
    
    # 2. Test batch inference
    mock_df = pd.DataFrame([
        [300.0, 310.0, 1500.0, 40.0, 50.0],
        [305.0, 315.0, 1300.0, 75.0, 200.0]
    ], columns=Config.FEATURE_COLUMNS)
    
    predictions_df = service.predict_batch(mock_df)
    assert "Prediction" in predictions_df.columns
    assert len(predictions_df) == 2
    assert predictions_df["Prediction"].iloc[0] in ["✅ Safe", "⚠️ Failure"]

def test_dashboard_service():
    """Validates metrics resolution and aggregates within DashboardService."""
    service = DashboardService()
    
    stats = service.get_machine_stats()
    assert "total" in stats
    assert "healthy" in stats
    assert "failures" in stats
    assert stats["total"] == stats["healthy"] + stats["failures"]
    
    metrics = service.get_model_performance_metrics()
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_simulation_service():
    """Validates simulation service telemetry values and anomaly injections."""
    service = SimulationService()
    telemetry = service.generate_sensor_telemetry()
    
    assert "air_temp" in telemetry
    assert "speed" in telemetry
    assert "torque" in telemetry
    assert "tool_wear" in telemetry
    assert "prediction" in telemetry
    assert telemetry["status"] in ["Healthy", "Warning", "Critical"]
    assert isinstance(telemetry["anomaly_injected"], bool)

def test_report_service():
    """Validates PDF report compilation and timestamped file registration."""
    service = ReportService()
    pdf_path = service.generate_pdf_report()
    
    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    # Clean up generated test report file
    try:
        os.remove(pdf_path)
    except OSError:
        pass

def test_database_service():
    """Validates persistent database CRUD workflows using SQLAlchemy and DatabaseService in isolation."""
    from backend.app.app import create_app
    from backend.app.extensions import db
    from backend.app.services.database_service import DatabaseService
    from backend.app.models.prediction import Prediction
    from backend.app.models.report import Report
    from backend.app.models.log import Log
    
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    
    with app.app_context():
        # Re-create all tables in memory
        db.create_all()
        
        # 1. Test Prediction persistence
        mock_features = [302.5, 312.1, 1480.0, 42.1, 12.0]
        pred_label = "✅ Safe"
        confidence = 94.5
        status_str = "SAFE"
        
        entry = DatabaseService.save_prediction(mock_features, pred_label, confidence, status_str)
        assert entry.id is not None
        assert entry.air_temperature == 302.5
        assert entry.prediction == pred_label
        
        recent = DatabaseService.get_recent_predictions(limit=5)
        assert len(recent) >= 1
        assert recent[0]["result"] == pred_label
        assert recent[0]["status"] == "SAFE"
        
        # 2. Test Report persistence
        rep = DatabaseService.save_report_metadata("test_report.pdf", "/path/to/test_report.pdf")
        assert rep.id is not None
        assert rep.report_name == "test_report.pdf"
        
        # 3. Test Log persistence
        log = DatabaseService.save_log_entry("INFO", "Database test running")
        assert log.id is not None
        assert log.log_type == "INFO"
        assert log.message == "Database test running"
