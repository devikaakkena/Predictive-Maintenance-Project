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
