from flask import Blueprint, jsonify
from backend.app.services.simulation_service import SimulationService
from backend.app.utils.logger import simulation_logger, errors_logger

simulation_bp = Blueprint("simulation", __name__, url_prefix="/api")
simulation_service = SimulationService()

@simulation_bp.route("/simulation-stream", methods=["GET"])
def get_simulation_stream():
    """
    Exposes a JSON endpoint returning real-time simulated telemetry
    and automated classification labels.
    
    Returns:
        Response: The telemetry event payload in JSON format.
    """
    simulation_logger.info("Serving dynamic real-time operational sensor telemetry event...")
    try:
        telemetry = simulation_service.generate_sensor_telemetry()
        simulation_logger.info(
            f"Simulated Event - AirTemp: {telemetry['air_temp']}K, Speed: {telemetry['speed']}rpm, "
            f"Torque: {telemetry['torque']}Nm, ToolWear: {telemetry['tool_wear']}min -> "
            f"Prediction: {telemetry['prediction']}, Status: {telemetry['status']}"
        )
        return jsonify(telemetry)
    except Exception as e:
        errors_logger.error(f"Failed to generate sensor telemetry event: {str(e)}")
        return jsonify({"error": str(e)}), 500

