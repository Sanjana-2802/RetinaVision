"""
Comprehensive Unit and Integration Tests for the Four Core Ophthalmic Pillars:
1. Multi-Disease Identification & Screening (GhostNet + Grad-CAM)
2. Disease Severity Grading (ConvNeXt 5-Stage ICDR Hybrid Score)
3. Pixel-Level Lesion Segmentation (U-Net Multi-Channel Binary Masks)
4. Anatomical Landmark Localization (Optic Disc CDR & Fovea CSME Risk)
5. REST API Endpoints (/api/presets, /api/analyze, /api/health)
"""

import os
import sys
import cv2
import numpy as np
import torch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.ml.ghostnet_screening import MultiDiseaseGhostNet, GhostNetGradCAM, SCREENING_DISEASES
from app.ml.convnext_dr_grader import ConvNeXtDRGrader, HybridSeverityEngine, DR_STAGES
from app.ml.unet_segmentation import UNetMultiLesion, RetinalLesionSegmenter
from app.ml.landmarks import RetinalLandmarkDetector
from app.ml.presets import PRESET_DIR, generate_preset_images
from app.ml.model import ClinicalDiagnosticsEngine


@pytest.fixture(scope="session")
def client():
    generate_preset_images()
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------------
# Pillar 1 Tests: GhostNet Multi-Disease Screening & Grad-CAM
# -------------------------------------------------------------

def test_ghostnet_architecture_and_gradcam():
    model = MultiDiseaseGhostNet()
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Forward pass
    logits = model(dummy_input)
    assert logits.shape == (1, len(SCREENING_DISEASES))

    # Grad-CAM forward & backward
    cam_engine = GhostNetGradCAM(model, model.target_layer)
    cam = cam_engine.generate(dummy_input, class_idx=0)
    assert cam.shape[0] > 0 and cam.shape[1] > 0
    assert 0.0 <= cam.min() and cam.max() <= 1.0

    # Heatmap overlay
    dummy_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay, cam_resized = cam_engine.generate_heatmap_overlay(dummy_rgb, cam)
    assert overlay.shape == (512, 512, 3)
    assert cam_resized.shape == (512, 512)


# -------------------------------------------------------------
# Pillar 2 Tests: ConvNeXt 5-Stage DR Severity Grading & Hybrid Score
# -------------------------------------------------------------

def test_convnext_dr_grading_and_hybrid_scoring():
    grader = ConvNeXtDRGrader(pretrained=True)
    grader.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    logits = grader(dummy_input)
    assert logits.shape == (1, 5)

    engine = HybridSeverityEngine(model=grader)
    
    # Test Grade 0 (Normal)
    metrics_normal = {"ma_count": 0, "he_count": 0, "he_area_pct": 0.0, "ex_count": 0, "se_count": 0, "fovea_dist_dd": 99.0}
    res_normal = engine.grade_scan(dummy_input, metrics_normal, filename="sample_normal.jpg")
    assert res_normal["dr_grade"] == 0
    assert "Grade 0" in res_normal["stage_name"]

    # Test Grade 1 (Mild - MAs only)
    metrics_mild = {"ma_count": 10, "he_count": 0, "he_area_pct": 0.0, "ex_count": 0, "se_count": 0, "fovea_dist_dd": 99.0}
    res_mild = engine.grade_scan(dummy_input, metrics_mild, filename="sample_dr_mild.jpg")
    assert res_mild["dr_grade"] == 1
    assert "Mild" in res_mild["short_name"]

    # Test Grade 2 (Moderate - MAs + HEs + EX)
    metrics_mod = {"ma_count": 25, "he_count": 10, "he_area_pct": 0.5, "ex_count": 15, "se_count": 0, "fovea_dist_dd": 1.2}
    res_mod = engine.grade_scan(dummy_input, metrics_mod, filename="sample_dr_moderate.jpg")
    assert res_mod["dr_grade"] == 2
    assert "Moderate" in res_mod["short_name"]

    # Test Grade 3 (Severe - 4-quadrant hemorrhages)
    metrics_sev = {"ma_count": 55, "he_count": 28, "he_area_pct": 1.8, "ex_count": 35, "se_count": 4, "fovea_dist_dd": 0.8}
    res_sev = engine.grade_scan(dummy_input, metrics_sev, filename="sample_dr_severe.jpg")
    assert res_sev["dr_grade"] == 3
    assert "Severe" in res_sev["short_name"]

    # Test Grade 4 (Proliferative DR)
    metrics_pdr = {"ma_count": 60, "he_count": 40, "he_area_pct": 4.2, "ex_count": 45, "se_count": 3, "fovea_dist_dd": 0.2, "neovascularization": True}
    res_pdr = engine.grade_scan(dummy_input, metrics_pdr, filename="sample_dr_proliferative.jpg")
    assert res_pdr["dr_grade"] == 4
    assert "Proliferative" in res_pdr["short_name"]


# -------------------------------------------------------------
# Pillar 3 Tests: U-Net Multi-Lesion Segmentation
# -------------------------------------------------------------

def test_unet_lesion_segmentation():
    unet = UNetMultiLesion()
    unet.eval()
    x = torch.randn(1, 3, 256, 256)
    out = unet(x)
    assert out.shape == (1, 4, 256, 256)  # 4 channels: [MA, HE, EX, SE]

    segmenter = RetinalLesionSegmenter(model=unet)
    dummy_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(dummy_bgr, (256, 256), 220, (25, 60, 180), -1)
    # Add lesion dots
    cv2.circle(dummy_bgr, (310, 240), 2, (10, 15, 80), -1)
    cv2.circle(dummy_bgr, (270, 230), 4, (120, 240, 255), -1)

    retinal_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(retinal_mask, (256, 256), 220, 255, -1)

    res = segmenter.segment_image(x, dummy_bgr, dummy_bgr, retinal_mask, "app/static/uploads", prefix="test_unet")
    assert "mask_filenames" in res
    assert "ma" in res["mask_filenames"]
    assert "he" in res["mask_filenames"]
    assert "ex" in res["mask_filenames"]
    assert "se" in res["mask_filenames"]
    assert "composite_filename" in res
    assert os.path.exists(os.path.join("app/static/uploads", res["composite_filename"]))


# -------------------------------------------------------------
# Pillar 4 Tests: Anatomical Landmarks & CSME Risk
# -------------------------------------------------------------

def test_anatomical_landmarks_and_csme():
    detector = RetinalLandmarkDetector()
    dummy_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(dummy_bgr, (256, 256), 230, (20, 60, 185), -1)
    # OD at (145, 256)
    cv2.circle(dummy_bgr, (145, 256), 42, (120, 205, 245), -1)
    cv2.circle(dummy_bgr, (145, 256), 16, (160, 230, 255), -1)
    # Fovea at (330, 260)
    cv2.circle(dummy_bgr, (330, 260), 30, (10, 35, 120), -1)

    retinal_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(retinal_mask, (256, 256), 230, 255, -1)

    od_info = detector.detect_optic_disc(dummy_bgr, dummy_bgr, retinal_mask)
    assert od_info["x"] > 0 and od_info["y"] > 0
    assert 0.1 <= od_info["cup_to_disc_ratio"] <= 1.0

    fovea_info = detector.pinpoint_fovea(dummy_bgr, dummy_bgr, od_info, retinal_mask)
    assert fovea_info["x"] > 0 and fovea_info["y"] > 0
    assert "Eye" in fovea_info["eye_laterality"]

    # CSME threat test with fake exudate mask near fovea
    ex_mask_near = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(ex_mask_near, (fovea_info["x"] + 10, fovea_info["y"] + 10), 3, 255, -1)
    csme_near = detector.evaluate_macular_threat(fovea_info, od_info, {"ex": ex_mask_near})
    assert "High Risk" in csme_near["csme_risk"]
    assert csme_near["nearest_ex_dist_dd"] <= 0.33


# -------------------------------------------------------------
# REST API Endpoints Tests
# -------------------------------------------------------------

def test_api_presets_endpoint(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 8


def test_api_analyze_with_preset(client):
    res = client.post("/api/analyze", data={"preset_filename": "sample_dr_moderate.jpg"})
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_disease"] == "Diabetic Retinopathy"
    assert "dr_grade" in data
    assert "lesion_stats" in data
    assert "optic_disc" in data
    assert "fovea" in data
    assert "csme" in data
    assert "image_url" in data
    assert "mask_composite_url" in data
