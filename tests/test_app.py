import os
import sys

# Prepend current directory to sys.path so local app package is resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.preprocessor import preprocess_fundus_image
from app.ml.presets import generate_preset_images, PRESET_DIR
from app.ml.model import ClinicalDiagnosticsEngine
from app.services.recommendations import get_health_recommendations
from app.services.hospitals import get_recommended_hospitals
from app.services.progression import analyze_patient_progression
from app.services.pdf_report import generate_medical_report_pdf
from app.database.models import Patient, ScanRecord

@pytest.fixture(scope="session")
def client():
    # Setup presets before running client tests
    generate_preset_images()
    with TestClient(app) as c:
        yield c

def test_health_check_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["batch"] == "33"

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "RetinaCare" in response.text
    assert "BATCH NO: 33" in response.text

def test_diagnose_page(client):
    response = client.get("/diagnose")
    assert response.status_code == 200
    assert "Retinal Fundus Image Analysis" in response.text

def test_patients_page(client):
    response = client.get("/patients")
    assert response.status_code == 200
    assert "Patient Directory" in response.text

def test_hospitals_page(client):
    response = client.get("/hospitals")
    assert response.status_code == 200
    assert "Eye Specialists" in response.text

def test_about_page(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert "23551A42B3" in response.text
    assert "23551A4275" in response.text
    assert "24555A4207" in response.text

def test_ml_preprocessing_and_inference():
    preset_sample = os.path.join(PRESET_DIR, "sample_normal.jpg")
    assert os.path.exists(preset_sample), "Sample preset should exist"

    tensor, orig_rgb, enh_rgb = preprocess_fundus_image(preset_sample)
    assert tensor.shape == (1, 3, 256, 256)
    assert orig_rgb.shape == (512, 512, 3)
    assert enh_rgb.shape == (512, 512, 3)

    engine = ClinicalDiagnosticsEngine.get_instance()
    gradcam_output = os.path.join("app/static/uploads", "test_gradcam_out.jpg")
    res = engine.analyze_scan(tensor, orig_rgb, enh_rgb, "sample_normal.jpg", output_gradcam_path=gradcam_output)

    assert "predicted_disease" in res
    assert "severity" in res
    assert "confidence" in res
    assert "biomarkers" in res
    assert "explanation" in res
    assert os.path.exists(gradcam_output)

def test_recommendation_engine():
    rec_dr = get_health_recommendations("Diabetic Retinopathy", "Severe")
    assert rec_dr["hospital_recommendation"] is True
    assert len(rec_dr["nutrients"]) > 0
    assert "meal_plan" in rec_dr
    assert len(rec_dr["foods_to_avoid"]) > 0

    rec_norm = get_health_recommendations("Normal", "Normal")
    assert rec_norm["hospital_recommendation"] is False
    assert "20-20-20 rule" in " ".join(rec_norm["eye_care_routine"])

def test_hospitals_filtering():
    all_hosps = get_recommended_hospitals()
    assert len(all_hosps) >= 3
    glaucoma_hosps = get_recommended_hospitals(disease="Glaucoma")
    assert len(glaucoma_hosps) > 0

def test_pdf_generation():
    p_info = {
        "name": "Jane Doe",
        "patient_id": "PAT-TEST-99",
        "age": 52,
        "gender": "Female",
        "contact": "9876543210",
        "medical_history": "Hypertension"
    }
    s_info = {
        "predicted_disease": "Glaucoma",
        "severity": "Moderate",
        "confidence": 0.91,
        "risk_score": 62.0,
        "cup_to_disc_ratio": 0.68,
        "macular_integrity": 85.0,
        "exudate_density": 5.0,
        "vessel_tortuosity": 1.1,
        "scan_date": "23 Sep 2026"
    }
    rec = get_health_recommendations("Glaucoma", "Moderate")
    img_sample = os.path.join(PRESET_DIR, "sample_glaucoma.jpg")
    out_pdf = generate_medical_report_pdf(p_info, s_info, img_sample, img_sample, rec, "test_generated_unit.pdf")
    assert os.path.exists(out_pdf)
    assert os.path.getsize(out_pdf) > 1000  # Non-trivial PDF generated
