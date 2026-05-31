import time
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Resolve absolute paths
BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

def save_classification_report_txt(
    metrics_summary: str,
    class_report_str: str,
    save_dir: Path = REPORTS_DIR
) -> Path:
    """
    Saves details of model classification validation reports to a text file.
    
    Args:
        metrics_summary (str): Summarized metrics key-values.
        class_report_str (str): Full Scikit-Learn classification report metrics table.
        save_dir (Path): Output directory.
        
    Returns:
        Path: Saved file path.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"classification_report_{timestamp}.txt"
    save_path = save_dir / filename
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("==================================================================\n")
        f.write("       PREDICTIVE MAINTENANCE SYSTEM - DETAILED EVALUATION REPORT  \n")
        f.write("==================================================================\n\n")
        f.write("Validation Metrics Summary:\n")
        f.write("------------------------------------------------------------------\n")
        f.write(metrics_summary)
        f.write("\n\nClassification Performance Details:\n")
        f.write("------------------------------------------------------------------\n")
        f.write(class_report_str)
        f.write("\n==================================================================\n")
        
    logger.info(f"Successfully generated detailed metrics validation report: {save_path}")
    return save_path
