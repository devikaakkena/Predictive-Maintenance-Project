from flask import Blueprint, jsonify
import logging
from backend.app.services.simulation_service import SimulationService

# Configure logger
logger = logging.getLogger(__name__)

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
    logger.info("Serving dynamic real-time operational sensor telemetry event...")
    try:
        telemetry = simulation_service.generate_sensor_telemetry()
        return jsonify(telemetry)
    except Exception as e:
        logger.error(f"Failed to generate sensor telemetry event: {str(e)}")
        return jsonify({"error": str(e)}), 500
