from flask_sqlalchemy import SQLAlchemy

# Instantiate SQLAlchemy ORM globally to avoid circular blueprints/models imports
db = SQLAlchemy()
