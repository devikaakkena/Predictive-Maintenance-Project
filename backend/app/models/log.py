from datetime import datetime
from backend.app.extensions import db

class Log(db.Model):
    """
    ORM Model representing persistent structured system events and background tasks.
    """
    __tablename__ = "logs"
    
    id = db.Column(db.Integer, primary_key=True, index=True)
    log_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Converts the database ORM fields into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "log_type": self.log_type,
            "message": self.message,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else ""
        }
