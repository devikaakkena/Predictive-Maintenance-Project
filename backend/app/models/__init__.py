from backend.app.models.prediction import Prediction
from backend.app.models.report import Report
from backend.app.models.log import Log

# Expose models cleanly to app bootstrap initialization
__all__ = ["Prediction", "Report", "Log"]
