from flask import Blueprint, send_file
import logging
from backend.app.services.report_service import ReportService

# Configure logger
logger = logging.getLogger(__name__)

report_bp = Blueprint("report", __name__)
report_service = ReportService()

@report_bp.route("/analysis/download-pdf", methods=["GET"])
def download_pdf_report():
    """
    Compiles and serves the dynamically generated Predictive Maintenance PDF Report.
    
    Returns:
        Response: The downloadable PDF file attachment.
    """
    logger.info("Inbound browser request received: Downloading dynamic operational PDF report...")
    try:
        pdf_path = report_service.generate_pdf_report()
        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_path.name
        )
    except Exception as e:
        logger.error(f"Failed to compile and serve PDF report: {str(e)}")
        return f"<h3>❌ Failed to generate operations analysis report: {str(e)}</h3>", 500
