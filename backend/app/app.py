import os
from flask import Flask
from backend.app.config.settings import Config
from backend.app.extensions import db

def create_app(config_class=Config):
    # Initialize the Flask application
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure the standard Flask instance directory exists on disk for SQLite db
    # Flask sets app.instance_path to '<project_root>/instance' by default
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Initialize SQLAlchemy cleanly
    db.init_app(app)
    
    # Import and register Blueprints to avoid circular imports
    from backend.app.routes.dashboard_routes import dashboard_bp
    from backend.app.routes.prediction_routes import prediction_bp
    from backend.app.routes.analysis_routes import analysis_bp
    from backend.app.routes.api_routes import api_bp
    from backend.app.routes.report_routes import report_bp
    from backend.app.routes.simulation_routes import simulation_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(simulation_bp)
    
    # Automatically generate SQLite database tables inside application context
    with app.app_context():
        # Import models to ensure they are registered in SQLAlchemy metadata
        import backend.app.models  # noqa
        db.create_all()
    
    return app

# Expose app for WSGI servers
app = create_app()