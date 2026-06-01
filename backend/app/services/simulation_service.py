import random
from backend.app.services.prediction_service import PredictionService
from backend.app.utils.logger import simulation_logger


# Persistent simulation state to model tool wear accumulation and resetting
_sim_state = {
    "tool_wear": 0.0,
    "runs": 0
}

class SimulationService:
    def __init__(self):
        self.prediction_service = PredictionService()

    def generate_sensor_telemetry(self) -> dict:
        """
        Generates realistic industrial sensor telemetry values and runs
        real-time inference using the pre-trained optimal ML classifier.
        
        Physical boundaries (Normal Operating conditions):
        - Air temperature [K]: 297.0K - 302.0K
        - Process temperature [K]: Air temperature + ~10K
        - Rotational speed [rpm]: 1400 - 1600 rpm (inversely proportional to torque)
        - Torque [Nm]: 38 - 48 Nm
        - Tool wear [min]: 0 - 240 min (wear accumulates and resets)
        """
        global _sim_state
        _sim_state["runs"] += 1
        
        # 1. Accumulate tool wear, trigger a dynamic reset after 240 minutes
        _sim_state["tool_wear"] += random.uniform(1.0, 3.5)
        if _sim_state["tool_wear"] > 240.0:
            simulation_logger.info("Simulation: Tool wear limit exceeded. Simulating cutting head replacements...")
            _sim_state["tool_wear"] = 0.0
            
        tool_wear = round(_sim_state["tool_wear"], 1)
        
        # 2. Simulate standard industrial fluctuations
        air_temp = round(random.uniform(297.0, 302.0), 1)
        process_temp = round(air_temp + random.uniform(9.5, 11.5), 1)
        
        # Normal speed and torque have a strong physical inverse correlation
        speed = round(random.uniform(1420.0, 1580.0), 1)
        torque = round(random.uniform(39.0, 47.0), 1)
        
        # 3. Periodic anomaly injection (e.g. 8% baseline, or up to 25% when tool wear is critical)
        anomaly_injected = False
        
        if random.random() < 0.08 or (tool_wear > 190.0 and random.random() < 0.25):
            anomaly_injected = True
            simulation_logger.warning("Simulation: Anomaly spike injected into telemetry stream!")
            # Simulate high-load friction failures (torque spike, rotational speed drop)
            torque = round(random.uniform(62.0, 78.0), 1)
            speed = round(random.uniform(980.0, 1150.0), 1)
            
        # 4. Execute Real-Time Machine Learning Inference & Dynamic DB Persistence!
        features = [air_temp, process_temp, speed, torque, tool_wear]
        pred_detail = self.prediction_service.predict_single_detailed(features)
        
        prediction = pred_detail["result"]
        db_status = pred_detail["status"]
        
        # Map DB status (SAFE, WARNING, CRITICAL) to simulation status (Healthy, Warning, Critical) for JS client
        status = "Healthy"
        if db_status == "CRITICAL":
            status = "Critical"
        elif db_status == "WARNING":
            status = "Warning"
            
        return {
            "air_temp": air_temp,
            "process_temp": process_temp,
            "speed": speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "prediction": prediction,
            "status": status,
            "anomaly_injected": anomaly_injected
        }
