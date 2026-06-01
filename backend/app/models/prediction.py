from datetime import datetime
from backend.app.extensions import db

class Prediction(db.Model):
    """
    ORM Model representing persistent machine inference runs and telemetry checks.
    """
    __tablename__ = "predictions"
    
    id = db.Column(db.Integer, primary_key=True, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    air_temperature = db.Column(db.Float, nullable=False)
    process_temperature = db.Column(db.Float, nullable=False)
    rotational_speed = db.Column(db.Float, nullable=False)
    torque = db.Column(db.Float, nullable=False)
    tool_wear = db.Column(db.Float, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    machine_status = db.Column(db.String(50), nullable=False)

    def to_dict(self) -> dict:
        """
        Converts the database ORM fields into a JSON-compatible schema 
        matching standard template values expected by predictions.html.
        """
        return {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else "",
            "sensor_summary": f"Air: {self.air_temperature:.1f}K | Speed: {self.rotational_speed:.0f}rpm | Torque: {self.torque:.1f}Nm",
            "result": self.prediction,
            "confidence": int(self.confidence_score),
            "status": self.machine_status,
            "status_color": "danger" if self.machine_status == "CRITICAL" else ("warning" if self.machine_status == "WARNING" else "success")
        }
