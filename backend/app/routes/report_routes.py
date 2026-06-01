import os
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


@report_bp.route("/download-report/latest", methods=["GET"])
def download_latest_report():
    """
    Dynamically fetches and serves the latest compiled PDF report from database.
    If no report exists in DB or file is missing, compiles a new report on-the-fly.
    
    Returns:
        Response: The latest PDF report file attachment.
    """
    app_logger.info("Inbound browser request: Downloading latest compiled PDF report...")
    try:
        from backend.app.models.report import Report
        latest = Report.query.order_by(Report.generated_at.desc()).first()
        
        # Check if latest record exists and the physical PDF file is present
        if latest and os.path.exists(latest.file_path):
            pdf_path = latest.file_path
            app_logger.info(f"Serving cached latest PDF report from database: {pdf_path}")
        else:
            # Fallback to dynamic generation if database is empty or file is deleted
            app_logger.warning("Latest report file or database entry not found. Compiling a new report...")
            pdf_path = str(report_service.generate_pdf_report())
            
        from pathlib import Path
        path_obj = Path(pdf_path)
        return send_file(
            str(path_obj),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=path_obj.name
        )
    except Exception as e:
        errors_logger.error(f"Failed to download latest PDF report: {str(e)}")
        return f"<h3>❌ Failed to download latest operations analysis report: {str(e)}</h3>", 500


@report_bp.route("/download-report/<int:report_id>", methods=["GET"])
def download_report_by_id(report_id):
    """
    Serves a specific PDF report from database using its unique report ID.
    Features robust path checking to prevent directory traversal attacks.
    
    Args:
        report_id (int): The unique ID of the report inside SQLite table.
        
    Returns:
        Response: The requested PDF report file attachment.
    """
    app_logger.info(f"Inbound browser request: Downloading PDF report by ID: {report_id}")
    try:
        from backend.app.extensions import db
        from backend.app.models.report import Report
        report_entry = db.session.get(Report, report_id)
        if not report_entry:
            app_logger.warning(f"Failed download: Report ID {report_id} not registered in database.")
            return "<h3>❌ Error: Report record not found in database.</h3>", 404
            
        # Secure path validation
        pdf_path = os.path.abspath(report_entry.file_path)
        from backend.app.config.settings import Config
        base_reports_dir = os.path.abspath(Config.REPORT_OUTPUT_PATH)
        
        # Validate that the file is strictly contained within the outputs/reports folder
        if not pdf_path.startswith(base_reports_dir):
            errors_logger.warning(f"Security Alert: Directory traversal attempt blocked for path: {pdf_path}")
            return "<h3>❌ Access Denied: Invalid report path.</h3>", 403
            
        if not os.path.exists(pdf_path):
            app_logger.warning(f"Failed download: Physical report file does not exist at path: {pdf_path}")
            return "<h3>❌ Error: Report PDF file does not exist on disk.</h3>", 404
            
        from pathlib import Path
        path_obj = Path(pdf_path)
        return send_file(
            str(path_obj),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=path_obj.name
        )
    except Exception as e:
        errors_logger.error(f"Failed to download PDF report by ID {report_id}: {str(e)}")
        return f"<h3>❌ Failed to download operations analysis report: {str(e)}</h3>", 500
