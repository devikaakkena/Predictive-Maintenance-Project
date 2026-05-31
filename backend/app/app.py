from flask import Flask
from backend.app.config.settings import Config

def create_app(config_class=Config):
    # Initialize the Flask application
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Import and register Blueprints to avoid circular imports
    from backend.app.routes.dashboard_routes import dashboard_bp
    from backend.app.routes.prediction_routes import prediction_bp
    from backend.app.routes.analysis_routes import analysis_bp
    from backend.app.routes.api_routes import api_bp
    
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(api_bp)
    
    return app

# Expose app for WSGI servers
app = create_app()