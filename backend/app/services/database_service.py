from datetime import datetime
from backend.app.extensions import db
from backend.app.models.prediction import Prediction
from backend.app.models.report import Report
from backend.app.models.log import Log
from backend.app.utils.logger import app_logger, errors_logger

class DatabaseService:
    """
    Centralized Database Service layer encapsulating clean CRUD workflows 
    for predictions, compiled PDF reports, and structured operational events logs.
    """
    
    @staticmethod
    def save_prediction(features: list, prediction: str, confidence_score: float, machine_status: str) -> Prediction:
        """
        Saves an inference run record persistently to the SQLite database.
        
        Args:
            features (list): [air_temp, process_temp, speed, torque, tool_wear]
            prediction (str): Formatted outcome label (e.g. "✅ Safe", "⚠️ Failure")
            confidence_score (float): Confidence score or probability percentage (0-100)
            machine_status (str): Operational status badge text ("SAFE", "WARNING", "CRITICAL")
            
        Returns:
            Prediction: The saved model database entry.
        """
        try:
            air_temp, process_temp, speed, torque, tool_wear = features
            entry = Prediction(
                air_temperature=float(air_temp),
                process_temperature=float(process_temp),
                rotational_speed=float(speed),
                torque=float(torque),
                tool_wear=float(tool_wear),
                prediction=prediction,
                confidence_score=float(confidence_score),
                machine_status=machine_status
            )
            db.session.add(entry)
            db.session.commit()
            app_logger.info(f"Database: Persisted prediction run successfully (ID: {entry.id}, status: {machine_status}).")
            return entry
        except Exception as e:
            db.session.rollback()
            errors_logger.error(f"Database Error: Failed to save prediction entry: {str(e)}")
            raise e

    @staticmethod
    def get_recent_predictions(limit: int = 10) -> list:
        """
        Retrieves the latest predictions from the database ordered by timestamp desc.
        
        Returns:
            list: List of dictionary representations compatible with UI contexts.
        """
        try:
            entries = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
            return [e.to_dict() for e in entries]
        except Exception as e:
            errors_logger.error(f"Database Error: Failed to query recent predictions: {str(e)}")
            return []

    @staticmethod
    def save_report_metadata(report_name: str, file_path: str) -> Report:
        """
        Saves compiled PDF report generation metadata persistently to the database.
        
        Args:
            report_name (str): Basename of the PDF file.
            file_path (str): Absolute file path to the compiled report.
            
        Returns:
            Report: The saved report database entry.
        """
        try:
            entry = Report(
                report_name=report_name,
                file_path=file_path
            )
            db.session.add(entry)
            db.session.commit()
            app_logger.info(f"Database: Persisted report metadata successfully (ID: {entry.id}, Name: {report_name}).")
            return entry
        except Exception as e:
            db.session.rollback()
            errors_logger.error(f"Database Error: Failed to save report metadata: {str(e)}")
            raise e

    @staticmethod
    def save_log_entry(log_type: str, message: str) -> Log:
        """
        Saves a structured log event record to the database logs table.
        
        Args:
            log_type (str): Category type of log ("INFO", "WARNING", "ERROR", "SIMULATION")
            message (str): Structured log message content.
            
        Returns:
            Log: The saved log database entry.
        """
        try:
            entry = Log(
                log_type=log_type,
                message=message
            )
            db.session.add(entry)
            db.session.commit()
            return entry
        except Exception as e:
            db.session.rollback()
            errors_logger.error(f"Database Error: Failed to save log entry: {str(e)}")
            raise e
