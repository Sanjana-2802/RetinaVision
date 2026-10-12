import os
import sys

# Prepend current directory to sys.path so local app package is resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.preprocessor import preprocess_fundus_complete
from app.ml.presets import generate_preset_images, PRESET_DIR
from app.ml.model import ClinicalDiagnosticsEngine
from app.services.pdf_report import generate_medical_report_pdf


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
    assert data["version"] == "2.0.0"
    assert "four_pillars" in data
    assert "public_benchmarks" in data


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Retinal Health" in response.text
    assert "SpectraRetina" in response.text


def test_diagnose_page(client):
    response = client.get("/diagnose")
    assert response.status_code == 200
    assert "Retinal Scan Intake" in response.text
    assert "Sample Retinal Test Scans" in response.text


def test_benchmarks_page(client):
    response = client.get("/benchmarks")
    assert response.status_code == 200
    assert "Benchmark Datasets" in response.text
    assert "IDRiD" in response.text
    assert "Messidor-2" in response.text
    assert "REFUGE" in response.text


def test_about_page(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert "Automated Multi-Lesion Segmentation" in response.text
    assert "GhostNet" in response.text
    assert "ConvNeXt" in response.text
    assert "U-Net" in response.text


def test_api_presets(client):
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 5


def test_diagnose_post_with_preset(client):
    # Test diagnosis execution using benchmark preset
    response = client.post(
        "/diagnose",
        data={
            "preset_filename": "sample_dr_moderate.jpg",
            "patient_name": "Test Subject #101",
            "patient_age": 58,
            "patient_gender": "Male",
            "eye_laterality": "Right Eye (OD)",
            "clinical_notes": "Benchmark validation"
        },
        follow_redirects=False
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]
    assert redirect_url.startswith("/result/")

    # Follow redirect to result page
    res_page = client.get(redirect_url)
    assert res_page.status_code == 200
    assert "Interactive Multi-Channel Visual Studio" in res_page.text
    assert "U-Net" in res_page.text
    assert "ConvNeXt" in res_page.text
    assert "GhostNet" in res_page.text


def test_pdf_generation():
    p_info = {
        "name": "Jane Doe",
        "patient_id": "PAT-TEST-99",
        "age": 52,
        "gender": "Female",
        "medical_history": "Type 2 Diabetes"
    }
    s_info = {
        "predicted_disease": "Diabetic Retinopathy",
        "severity": "Moderate",
        "confidence": 0.92,
        "risk_score": 65.0,
        "dr_grade": 2,
        "dr_stage_name": "Grade 2: Moderate Non-Proliferative DR",
        "hybrid_severity_score": 62.5,
        "ma_count": 8,
        "he_count": 5,
        "he_area_pct": 0.42,
        "ex_count": 6,
        "ex_area_pct": 0.35,
        "se_count": 1,
        "se_area_pct": 0.08,
        "cup_to_disc_ratio": 0.42,
        "fovea_dist_dd": 1.2,
        "csme_risk": "Moderate Risk",
        "eye_laterality": "Right Eye (OD)",
        "referral_urgency": "Timely Referral",
        "referral_timeline": "Within 1 - 3 Months"
    }
    orig_sample = os.path.join(PRESET_DIR, "sample_dr_moderate.jpg")
    output_pdf = "test_clinical_report.pdf"

    pdf_path = generate_medical_report_pdf(
        patient_info=p_info,
        scan_info=s_info,
        original_img_path=orig_sample,
        gradcam_img_path=orig_sample,
        output_filename=output_pdf
    )

    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000


def test_export_mask_bundle_and_clinical_json(client):
    # 1. First trigger a diagnostic scan
    response = client.post(
        "/diagnose",
        data={
            "preset_filename": "sample_dr_mild.jpg",
            "patient_name": "Export Test Subject",
            "patient_age": 45,
            "patient_gender": "Female",
            "eye_laterality": "Right Eye (OD)",
            "clinical_notes": "Export Verification"
        },
        follow_redirects=False
    )
    assert response.status_code == 303
    redirect_url = response.headers["location"]
    scan_id = int(redirect_url.split("/")[-1])

    # 2. Test Mask ZIP Export
    zip_resp = client.get(f"/api/export-masks/{scan_id}")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"
    assert len(zip_resp.content) > 500

    # 3. Test Clinical JSON Export
    json_resp = client.get(f"/api/export-json/{scan_id}")
    assert json_resp.status_code == 200
    data = json_resp.json()
    assert "header" in data
    assert "diagnostics" in data
    assert "lesion_segmentation_unet" in data
    assert "anatomical_landmarks" in data
    assert data["subject"]["name"] == "Export Test Subject"
