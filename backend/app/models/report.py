from datetime import datetime
from backend.app.extensions import db

class Report(db.Model):
    """
    ORM Model representing compiled operational PDF analysis reports metadata.
    """
    __tablename__ = "reports"
    
    id = db.Column(db.Integer, primary_key=True, index=True)
    report_name = db.Column(db.String(100), nullable=False)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)

    def to_dict(self) -> dict:
        """Converts the database ORM fields into a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "report_name": self.report_name,
            "generated_at": self.generated_at.strftime("%Y-%m-%d %H:%M:%S") if self.generated_at else "",
            "file_path": self.file_path
        }
