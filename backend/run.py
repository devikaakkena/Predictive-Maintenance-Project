import sys
import os

# Resolve the absolute path of the workspace root and add it to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Resolve the absolute path of the backend directory and add it to sys.path
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import the configured Flask application
from backend.app.app import app

if __name__ == "__main__":
    app.run(debug=True)
