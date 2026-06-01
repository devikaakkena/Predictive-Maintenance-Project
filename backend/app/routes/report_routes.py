from flask import Blueprint, send_file
from backend.app.services.report_service import ReportService
from backend.app.utils.logger import app_logger, errors_logger

report_bp = Blueprint("report", __name__)
report_service = ReportService()

@report_bp.route("/analysis/download-pdf", methods=["GET"])
def download_pdf_report():
    """
    Compiles and serves the dynamically generated Predictive Maintenance PDF Report.
    
    Returns:
        Response: The downloadable PDF file attachment.
    """
    app_logger.info("Inbound browser request received: Downloading dynamic operational PDF report...")
    try:
        pdf_path = report_service.generate_pdf_report()
        app_logger.info(f"Successfully compiled and serving operational PDF report at: {pdf_path}")
        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_path.name
        )
    except Exception as e:
        errors_logger.error(f"Failed to compile and serve PDF report: {str(e)}")
        return f"<h3>❌ Failed to generate operations analysis report: {str(e)}</h3>", 500

