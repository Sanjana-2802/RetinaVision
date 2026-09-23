import os
import shutil
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.models import Patient, ScanRecord
from app.database.db import get_db, init_db
from app.ml.preprocessor import preprocess_fundus_image
from app.ml.presets import generate_preset_images
from app.ml.model import ClinicalDiagnosticsEngine
from app.services.recommendations import get_health_recommendations
from app.services.hospitals import get_recommended_hospitals
from app.services.progression import analyze_patient_progression
from app.services.pdf_report import generate_medical_report_pdf

UPLOAD_DIR = "app/static/uploads"
REPORTS_DIR = "app/static/reports"
PRESET_DIR = "app/static/images/presets"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure directories, generate presets, init and seed database
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PRESET_DIR, exist_ok=True)
    generate_preset_images()
    init_db()
    # Pre-warm deep learning engine
    ClinicalDiagnosticsEngine.get_instance()
    yield

app = FastAPI(
    title="AI-Based Eye Disease Identification and Health Recommendation System",
    description="Batch No: 33 (23551A42B3, 23551A4275, 24555A4207)",
    version="2.0.0",
    lifespan=lifespan
)

# Static and Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request, db: Session = Depends(get_db)):
    patients_count = db.query(Patient).count()
    scans_count = db.query(ScanRecord).count()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "home",
            "patients_count": patients_count,
            "scans_count": scans_count
        }
    )

@app.get("/diagnose", response_class=HTMLResponse)
async def diagnose_page(request: Request, preset: str = None, patient_id: int = None, db: Session = Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.name.asc()).all()
    selected_patient = db.query(Patient).filter(Patient.id == patient_id).first() if patient_id else None
    return templates.TemplateResponse(
        request=request,
        name="diagnose.html",
        context={
            "active_page": "diagnose",
            "patients": patients,
            "selected_preset": preset,
            "selected_patient": selected_patient
        }
    )

@app.post("/diagnose")
async def process_diagnosis(
    request: Request,
    file: UploadFile = File(None),
    preset_filename: str = Form(None),
    existing_patient_id: str = Form(None),
    patient_name: str = Form("Anonymous Patient"),
    patient_age: int = Form(45),
    patient_gender: str = Form("Other"),
    medical_history: str = Form(""),
    db: Session = Depends(get_db)
):
    # 1. Determine or Create Patient
    patient = None
    if existing_patient_id and existing_patient_id.isdigit():
        patient = db.query(Patient).filter(Patient.id == int(existing_patient_id)).first()

    if not patient:
        # Create new patient record
        new_pid = f"PAT-{datetime.utcnow().strftime('%y%m')}-{uuid.uuid4().hex[:4].upper()}"
        patient = Patient(
            patient_id=new_pid,
            name=patient_name or "Anonymous Patient",
            age=patient_age or 45,
            gender=patient_gender or "Other",
            medical_history=medical_history or "None specified",
            created_at=datetime.utcnow()
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # 2. Handle Image Upload or Preset
    unique_prefix = f"scan_{uuid.uuid4().hex[:8]}"
    raw_img_name = f"{unique_prefix}_orig.jpg"
    preprocessed_name = f"{unique_prefix}_clahe.jpg"
    gradcam_name = f"{unique_prefix}_gradcam.jpg"

    raw_img_path = os.path.join(UPLOAD_DIR, raw_img_name)
    preprocessed_path = os.path.join(UPLOAD_DIR, preprocessed_name)
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_name)

    if file and file.filename:
        # Save uploaded file
        with open(raw_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        source_identifier = file.filename
    elif preset_filename:
        # Copy from presets
        src_preset = os.path.join(PRESET_DIR, preset_filename)
        if not os.path.exists(src_preset):
            # Fallback if missing
            generate_preset_images()
        shutil.copyfile(src_preset, raw_img_path)
        source_identifier = preset_filename
    else:
        raise HTTPException(status_code=400, detail="No fundus image uploaded or preset selected.")

    # 3. Preprocess Fundus Image
    tensor, orig_rgb, enh_rgb = preprocess_fundus_image(raw_img_path, output_path=preprocessed_path)

    # 4. Run PyTorch Diagnostic & Grad-CAM Inference
    engine = ClinicalDiagnosticsEngine.get_instance()
    results = engine.analyze_scan(
        input_tensor=tensor,
        original_rgb=orig_rgb,
        enhanced_rgb=enh_rgb,
        image_filename=source_identifier,
        output_gradcam_path=gradcam_path
    )

    # 5. Generate Personalized Nutrition & Care Recommendations
    recommendations = get_health_recommendations(
        results["predicted_disease"],
        results["severity"]
    )

    # 6. Generate Clinical PDF Medical Report
    pdf_filename = f"{unique_prefix}_report.pdf"
    patient_dict = {
        "name": patient.name,
        "patient_id": patient.patient_id,
        "age": patient.age,
        "gender": patient.gender,
        "contact": patient.contact,
        "medical_history": patient.medical_history
    }
    scan_dict = {
        "predicted_disease": results["predicted_disease"],
        "severity": results["severity"],
        "confidence": results["confidence"],
        "risk_score": results["risk_score"],
        "biomarkers": results["biomarkers"],
        "scan_date": datetime.utcnow().strftime("%d %b %Y, %I:%M %p")
    }
    generate_medical_report_pdf(
        patient_info=patient_dict,
        scan_info=scan_dict,
        original_img_path=raw_img_path,
        gradcam_img_path=gradcam_path,
        recommendations=recommendations,
        output_filename=pdf_filename
    )

    # 7. Persist to Database
    scan_record = ScanRecord(
        patient_id=patient.id,
        scan_date=datetime.utcnow(),
        image_filename=raw_img_name,
        preprocessed_filename=preprocessed_name,
        gradcam_filename=gradcam_name,
        pdf_report_filename=pdf_filename,
        predicted_disease=results["predicted_disease"],
        severity=results["severity"],
        confidence=results["confidence"],
        risk_score=results["risk_score"],
        cup_to_disc_ratio=results["biomarkers"]["cup_to_disc_ratio"],
        macular_integrity=results["biomarkers"]["macular_integrity"],
        exudate_density=results["biomarkers"]["exudate_density"],
        vessel_tortuosity=results["biomarkers"]["vessel_tortuosity"],
        notes=results["explanation"]["summary"]
    )
    db.add(scan_record)
    db.commit()
    db.refresh(scan_record)

    return RedirectResponse(url=f"/result/{scan_record.id}", status_code=303)

@app.get("/result/{scan_id}", response_class=HTMLResponse)
async def view_result(request: Request, scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found")
    
    patient = scan.patient
    recommendations = get_health_recommendations(scan.predicted_disease, scan.severity)
    nearby_hospitals = get_recommended_hospitals(scan.predicted_disease)
    
    engine = ClinicalDiagnosticsEngine.get_instance()
    biomarkers = {
        "cup_to_disc_ratio": scan.cup_to_disc_ratio,
        "macular_integrity": scan.macular_integrity,
        "exudate_density": scan.exudate_density,
        "vessel_tortuosity": scan.vessel_tortuosity
    }
    xai_explanation = engine._generate_xai_explanation(scan.predicted_disease, scan.severity, biomarkers)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "active_page": "diagnose",
            "scan": scan,
            "patient": patient,
            "recommendations": recommendations,
            "nearby_hospitals": nearby_hospitals,
            "xai_explanation": xai_explanation
        }
    )

@app.get("/patients", response_class=HTMLResponse)
async def patients_page(request: Request, db: Session = Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="patients.html",
        context={
            "active_page": "patients",
            "patients": patients
        }
    )

@app.post("/patient/new")
async def create_patient(
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    contact: str = Form(""),
    medical_history: str = Form(""),
    db: Session = Depends(get_db)
):
    new_pid = f"PAT-{datetime.utcnow().strftime('%y%m')}-{uuid.uuid4().hex[:4].upper()}"
    p = Patient(
        patient_id=new_pid,
        name=name,
        age=age,
        gender=gender,
        contact=contact,
        medical_history=medical_history,
        created_at=datetime.utcnow()
    )
    db.add(p)
    db.commit()
    return RedirectResponse(url="/patients", status_code=303)

@app.get("/patient/{patient_id}", response_class=HTMLResponse)
async def patient_progression_page(request: Request, patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    progression = analyze_patient_progression(patient, patient.scans)

    return templates.TemplateResponse(
        request=request,
        name="progression.html",
        context={
            "active_page": "patients",
            "patient": patient,
            "progression": progression
        }
    )

@app.get("/hospitals", response_class=HTMLResponse)
async def hospitals_page(request: Request, disease: str = None):
    hospitals = get_recommended_hospitals(disease=disease)
    return templates.TemplateResponse(
        request=request,
        name="hospitals.html",
        context={
            "active_page": "hospitals",
            "hospitals": hospitals,
            "selected_disease": disease
        }
    )

@app.get("/report/download/{scan_id}")
async def download_report(scan_id: int, db: Session = Depends(get_db)):
    scan = db.query(ScanRecord).filter(ScanRecord.id == scan_id).first()
    if not scan or not scan.pdf_report_filename:
        raise HTTPException(status_code=404, detail="Medical report not found")
    
    pdf_path = os.path.join(REPORTS_DIR, scan.pdf_report_filename)
    if not os.path.exists(pdf_path):
        # Regenerate on the fly if needed
        patient = scan.patient
        patient_dict = {
            "name": patient.name,
            "patient_id": patient.patient_id,
            "age": patient.age,
            "gender": patient.gender,
            "contact": patient.contact,
            "medical_history": patient.medical_history
        }
        scan_dict = {
            "predicted_disease": scan.predicted_disease,
            "severity": scan.severity,
            "confidence": scan.confidence,
            "risk_score": scan.risk_score,
            "cup_to_disc_ratio": scan.cup_to_disc_ratio,
            "macular_integrity": scan.macular_integrity,
            "exudate_density": scan.exudate_density,
            "vessel_tortuosity": scan.vessel_tortuosity,
            "scan_date": scan.scan_date.strftime("%d %b %Y, %I:%M %p")
        }
        recommendations = get_health_recommendations(scan.predicted_disease, scan.severity)
        orig_p = os.path.join(UPLOAD_DIR, scan.image_filename)
        grad_p = os.path.join(UPLOAD_DIR, scan.gradcam_filename)
        generate_medical_report_pdf(
            patient_info=patient_dict,
            scan_info=scan_dict,
            original_img_path=orig_p,
            gradcam_img_path=grad_p,
            recommendations=recommendations,
            output_filename=scan.pdf_report_filename
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"RetinaCare_Report_{scan.patient.patient_id}_Scan{scan.id}.pdf"
    )

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "active_page": "about"
        }
    )

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "batch": "33",
        "project": "AI-Based Eye Disease Identification and Health Recommendation System",
        "timestamp": datetime.utcnow().isoformat()
    }
