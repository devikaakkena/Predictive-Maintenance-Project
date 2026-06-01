import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Resolve absolute path of outputs/logs directory using pathlib
BASE_DIR = Path(__file__).resolve().parents[3]
LOGS_DIR = BASE_DIR / "outputs" / "logs"

# Dynamically construct logs folder on disk to guarantee compatibility
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Custom logging parameters
LOG_LEVEL_STR = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
MAX_BYTES = 10 * 1024 * 1024  # 10 Megabytes limit
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

def _setup_logger(name: str, log_filename: str, level=LOG_LEVEL) -> logging.Logger:
    """
    Sets up an isolated logger with a RotatingFileHandler to avoid duplicate outputs.
    
    Args:
        name (str): Unique name identifier for the logger instance.
        log_filename (str): Target filename inside outputs/logs/.
        level: Minimum logging level configuration.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent log bubbling duplication
    
    # Check handlers to avoid duplicate log entries upon web server hot-reloads
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)
        log_path = LOGS_DIR / log_filename
        
        # 1. Rotating File Handler
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 2. Console Stream Handler (development console visibility)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

# Centralized loggers definition
app_logger = _setup_logger("app_activity", "app.log")
predictions_logger = _setup_logger("predictions_activity", "predictions.log")
errors_logger = _setup_logger("errors_activity", "errors.log", logging.ERROR)
simulation_logger = _setup_logger("simulation_activity", "simulation.log")
