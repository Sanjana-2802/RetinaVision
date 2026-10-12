import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.ml.presets import generate_preset_images


@pytest.fixture(scope="module")
def client():
    generate_preset_images()
    with TestClient(app) as c:
        yield c


def test_full_diagnosis_flow_with_preset(client):
    """
    Test full end-to-end clinical workstation pipeline:
    POST preset -> CLAHE -> U-Net Segmentation -> ConvNeXt DR Grading ->
    Landmark Localization -> GhostNet Screening & Grad-CAM -> PDF Report -> Result Studio
    """
    form_data = {
        "preset_filename": "sample_dr_moderate.jpg",
        "patient_name": "Clinical Validation Subject",
        "patient_age": 60,
        "patient_gender": "Male",
        "eye_laterality": "Right Eye (OD)",
        "clinical_notes": "Type-2 Diabetes Mellitus evaluation"
    }

    response = client.post("/diagnose", data=form_data, follow_redirects=True)
    assert response.status_code == 200
    assert "Diagnostic Analysis & Clinical Findings" in response.text
    assert "Diabetic Retinopathy" in response.text
    assert "Interactive Multi-Channel Visual Studio" in response.text
    assert "1. Multi-Condition Screening" in response.text
    assert "2. Diabetic Retinopathy Staging" in response.text
    assert "3. Retinal Lesion Detection" in response.text
    assert "4. Optic Landmarks" in response.text
    assert "Download Clinical PDF Report" in response.text
