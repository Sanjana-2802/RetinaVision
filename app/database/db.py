import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.database.models import Base, Patient, ScanRecord

DATABASE_URL = "sqlite:///./app/database/eye_health.db"

# Ensure database directory exists
os.makedirs(os.path.dirname("app/database/eye_health.db"), exist_ok=True)

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def migrate_db_columns():
    """Dynamically add new columns to scan_records table if not present."""
    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(scan_records)"))
            existing_cols = {row[1] for row in result.fetchall()}
            if not existing_cols:
                return
            new_columns = [
                ("dr_grade", "INTEGER DEFAULT 0"),
                ("dr_stage_name", "VARCHAR(100)"),
                ("hybrid_severity_score", "FLOAT"),
                ("ma_count", "INTEGER DEFAULT 0"),
                ("he_count", "INTEGER DEFAULT 0"),
                ("he_area_pct", "FLOAT DEFAULT 0.0"),
                ("ex_count", "INTEGER DEFAULT 0"),
                ("ex_area_pct", "FLOAT DEFAULT 0.0"),
                ("se_count", "INTEGER DEFAULT 0"),
                ("se_area_pct", "FLOAT DEFAULT 0.0"),
                ("total_lesion_pixels", "INTEGER DEFAULT 0"),
                ("optic_disc_x", "INTEGER"),
                ("optic_disc_y", "INTEGER"),
                ("optic_disc_radius", "INTEGER"),
                ("fovea_x", "INTEGER"),
                ("fovea_y", "INTEGER"),
                ("fovea_dist_dd", "FLOAT"),
                ("csme_risk", "VARCHAR(100)"),
                ("eye_laterality", "VARCHAR(50) DEFAULT 'Right Eye (OD)'"),
                ("multi_disease_json", "TEXT"),
                ("mask_composite_filename", "VARCHAR(255)"),
                ("landmark_filename", "VARCHAR(255)"),
                ("mask_ma_filename", "VARCHAR(255)"),
                ("mask_he_filename", "VARCHAR(255)"),
                ("mask_ex_filename", "VARCHAR(255)"),
                ("mask_se_filename", "VARCHAR(255)"),
                ("redfree_filename", "VARCHAR(255)"),
                ("gradcam_turbo_filename", "VARCHAR(255)"),
                ("gradcam_jet_filename", "VARCHAR(255)"),
                ("gradcam_viridis_filename", "VARCHAR(255)"),
                ("microns_per_pixel", "FLOAT"),
                ("etdrs_json", "TEXT"),
                ("uncertainty_json", "TEXT"),
            ]
            for col, col_type in new_columns:
                if col not in existing_cols:
                    conn.execute(text(f"ALTER TABLE scan_records ADD COLUMN {col} {col_type}"))
            conn.commit()
        except Exception as e:
            print("Database migration notice:", e)

def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_db_columns()
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Patient).count() == 0:
            seed_demo_data(db)
    finally:
        db.close()

def seed_demo_data(db):
    """Seed sample patients with multi-visit historical scans for progression tracking."""
    # Patient 1: Diabetic Retinopathy Progression
    p1 = Patient(
        patient_id="PAT-2024-001",
        name="Ramesh Patel",
        age=58,
        gender="Male",
        contact="+91 98765 43210",
        email="ramesh.patel@example.com",
        medical_history="Type-2 Diabetes Mellitus (12 years), Mild Hypertension. HbA1c: 8.4%",
        created_at=datetime.utcnow() - timedelta(days=240)
    )
    db.add(p1)
    db.commit()
    db.refresh(p1)

    # 3 historical scans for Ramesh
    s1_1 = ScanRecord(
        patient_id=p1.id,
        scan_date=datetime.utcnow() - timedelta(days=240),
        image_filename="presets/sample_dr_mild.jpg",
        preprocessed_filename="presets/sample_dr_mild.jpg",
        gradcam_filename="presets/sample_dr_mild.jpg",
        predicted_disease="Diabetic Retinopathy",
        severity="Mild",
        confidence=0.88,
        risk_score=38.0,
        cup_to_disc_ratio=0.35,
        macular_integrity=92.0,
        exudate_density=14.5,
        vessel_tortuosity=1.12,
        notes="Baseline examination. Isolated microaneurysms detected in temporal periphery."
    )
    s1_2 = ScanRecord(
        patient_id=p1.id,
        scan_date=datetime.utcnow() - timedelta(days=120),
        image_filename="presets/sample_dr_moderate.jpg",
        preprocessed_filename="presets/sample_dr_moderate.jpg",
        gradcam_filename="presets/sample_dr_moderate.jpg",
        predicted_disease="Diabetic Retinopathy",
        severity="Moderate",
        confidence=0.92,
        risk_score=62.0,
        cup_to_disc_ratio=0.36,
        macular_integrity=84.0,
        exudate_density=32.0,
        vessel_tortuosity=1.28,
        notes="Follow-up scan: Increased dot-blot hemorrhages and early hard exudate clusters."
    )
    s1_3 = ScanRecord(
        patient_id=p1.id,
        scan_date=datetime.utcnow() - timedelta(days=15),
        image_filename="presets/sample_dr_severe.jpg",
        preprocessed_filename="presets/sample_dr_severe.jpg",
        gradcam_filename="presets/sample_dr_severe.jpg",
        predicted_disease="Diabetic Retinopathy",
        severity="Severe",
        confidence=0.96,
        risk_score=86.0,
        cup_to_disc_ratio=0.38,
        macular_integrity=72.0,
        exudate_density=58.5,
        vessel_tortuosity=1.45,
        notes="Multiple intraretinal microvascular abnormalities (IRMA) with confluent exudates near macula. Immediate vitreoretinal referral advised."
    )
    db.add_all([s1_1, s1_2, s1_3])

    # Patient 2: Glaucoma Patient
    p2 = Patient(
        patient_id="PAT-2024-002",
        name="Eleanor Vance",
        age=64,
        gender="Female",
        contact="+91 98451 22345",
        email="eleanor.vance@example.com",
        medical_history="Family history of Primary Open-Angle Glaucoma. Baseline IOP: 24 mmHg OD, 26 mmHg OS.",
        created_at=datetime.utcnow() - timedelta(days=180)
    )
    db.add(p2)
    db.commit()
    db.refresh(p2)

    s2_1 = ScanRecord(
        patient_id=p2.id,
        scan_date=datetime.utcnow() - timedelta(days=180),
        image_filename="presets/sample_glaucoma.jpg",
        preprocessed_filename="presets/sample_glaucoma.jpg",
        gradcam_filename="presets/sample_glaucoma.jpg",
        predicted_disease="Glaucoma",
        severity="Moderate",
        confidence=0.91,
        risk_score=68.0,
        cup_to_disc_ratio=0.68,
        macular_integrity=88.0,
        exudate_density=2.0,
        vessel_tortuosity=1.08,
        notes="Optic nerve head shows marked neuroretinal rim thinning and vertical cup elongation."
    )
    s2_2 = ScanRecord(
        patient_id=p2.id,
        scan_date=datetime.utcnow() - timedelta(days=30),
        image_filename="presets/sample_glaucoma.jpg",
        preprocessed_filename="presets/sample_glaucoma.jpg",
        gradcam_filename="presets/sample_glaucoma.jpg",
        predicted_disease="Glaucoma",
        severity="Severe",
        confidence=0.95,
        risk_score=84.0,
        cup_to_disc_ratio=0.76,
        macular_integrity=82.0,
        exudate_density=2.5,
        vessel_tortuosity=1.10,
        notes="Progression noted in CDR (0.68 -> 0.76). Recommend visual field Humphrey perimetry."
    )
    db.add_all([s2_1, s2_2])

    # Patient 3: Healthy Normal
    p3 = Patient(
        patient_id="PAT-2024-003",
        name="Aarav Sharma",
        age=29,
        gender="Male",
        contact="+91 91234 56789",
        email="aarav.s@example.com",
        medical_history="No systemic illnesses. Routine annual computer vision screening.",
        created_at=datetime.utcnow() - timedelta(days=45)
    )
    db.add(p3)
    db.commit()
    db.refresh(p3)

    s3_1 = ScanRecord(
        patient_id=p3.id,
        scan_date=datetime.utcnow() - timedelta(days=45),
        image_filename="presets/sample_normal.jpg",
        preprocessed_filename="presets/sample_normal.jpg",
        gradcam_filename="presets/sample_normal.jpg",
        predicted_disease="Normal",
        severity="Normal",
        confidence=0.98,
        risk_score=5.0,
        cup_to_disc_ratio=0.28,
        macular_integrity=98.5,
        exudate_density=0.0,
        vessel_tortuosity=1.02,
        notes="Healthy fundus. Clear macula, crisp optic disc margins, normal arteriovenous ratio."
    )
    db.add(s3_1)

    db.commit()
