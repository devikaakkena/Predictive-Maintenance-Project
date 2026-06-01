import pytest
from backend.app.app import create_app

@pytest.fixture
def client():
    """Sets up the Flask test client fixture for endpoints verification."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Validates that the landing dashboard loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "PREDMAINT" in html or "Dashboard" in html or "Healthy" in html

def test_analytics_route(client):
    """Validates that the analytics metrics page loads successfully."""
    response = client.get("/analysis")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Model Performance" in html or "Validation" in html or "Analytics" in html

def test_predictions_routes(client):
    """Validates that the manual inference form and submissions complete successfully."""
    # 1. Test GET request
    get_res = client.get("/predictions")
    assert get_res.status_code == 200
    html_get = get_res.data.decode("utf-8")
    assert "Telemetry" in html_get or "Predict" in html_get

    # 2. Test POST submission
    post_res = client.post("/predict", data={
        "air_temp": "300.0",
        "process_temp": "310.0",
        "speed": "1500.0",
        "torque": "40.0",
        "tool_wear": "50.0"
    })
    assert post_res.status_code == 200
    html_post = post_res.data.decode("utf-8")
    assert "Safe" in html_post or "Failure" in html_post


def test_api_routes(client):
    """Validates all REST API feeds returns successful JSON response codes."""
    # 1. Dashboard Metrics
    r1 = client.get("/api/dashboard-metrics")
    assert r1.status_code == 200
    assert "accuracy" in r1.json
    
    # 2. Model Comparisons
    r2 = client.get("/api/model-comparison")
    assert r2.status_code == 200
    assert "metrics" in r2.json
    
    # 3. Prediction Summary
    r3 = client.get("/api/prediction-summary")
    assert r3.status_code == 200
    assert "labels" in r3.json
    
    # 4. Feature Importance
    r4 = client.get("/api/feature-importance")
    assert r4.status_code == 200
    assert "features" in r4.json
    
    # 5. Live Simulation Stream
    r5 = client.get("/api/simulation-stream")
    assert r5.status_code == 200
    assert "air_temp" in r5.json
    assert "prediction" in r5.json

def test_pdf_report_download(client):
    """Validates that the operational report compiles and serves a PDF file."""
    response = client.get("/analysis/download-pdf")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert "attachment" in response.headers.get("Content-Disposition", "")

def test_download_report_routes(client):
    """Validates the new dynamic database-backed PDF report download endpoints."""
    # 1. Test latest route (compiling on the fly fallback)
    r_latest = client.get("/download-report/latest")
    assert r_latest.status_code == 200
    assert r_latest.mimetype == "application/pdf"
    
    # 2. Test ID download with missing ID (should yield 404)
    r_id_missing = client.get("/download-report/99999")
    assert r_id_missing.status_code == 404
