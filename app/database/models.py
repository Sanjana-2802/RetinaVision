from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    contact = Column(String(50), nullable=True)
    email = Column(String(100), nullable=True)
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    scans = relationship("ScanRecord", back_populates="patient", cascade="all, delete-orphan", order_by="ScanRecord.scan_date.asc()")

    def __repr__(self):
        return f"<Patient {self.patient_id} - {self.name}>"


class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    scan_date = Column(DateTime, default=datetime.utcnow)

    # Image file paths
    image_filename = Column(String(255), nullable=False)
    preprocessed_filename = Column(String(255), nullable=True)
    redfree_filename = Column(String(255), nullable=True)
    gradcam_filename = Column(String(255), nullable=True)
    gradcam_turbo_filename = Column(String(255), nullable=True)
    gradcam_jet_filename = Column(String(255), nullable=True)
    gradcam_viridis_filename = Column(String(255), nullable=True)
    landmark_filename = Column(String(255), nullable=True)
    mask_composite_filename = Column(String(255), nullable=True)
    mask_ma_filename = Column(String(255), nullable=True)
    mask_he_filename = Column(String(255), nullable=True)
    mask_ex_filename = Column(String(255), nullable=True)
    mask_se_filename = Column(String(255), nullable=True)
    pdf_report_filename = Column(String(255), nullable=True)

    # Clinical AI Predictions
    predicted_disease = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)  # Normal, Mild, Moderate, Severe, Proliferative / Advanced
    confidence = Column(Float, nullable=False)      # 0.0 to 1.0
    risk_score = Column(Float, nullable=False)      # 0 to 100

    # 1. Multi-Disease Identification & Screening (GhostNet)
    multi_disease_json = Column(Text, nullable=True)

    # 2. Disease Severity Grading (ConvNeXt + Hybrid)
    dr_grade = Column(Integer, default=0)            # 0 to 4
    dr_stage_name = Column(String(100), nullable=True)
    hybrid_severity_score = Column(Float, nullable=True)

    # 3. Pixel-Level Lesion Segmentation (U-Net)
    ma_count = Column(Integer, default=0)
    he_count = Column(Integer, default=0)
    he_area_pct = Column(Float, default=0.0)
    ex_count = Column(Integer, default=0)
    ex_area_pct = Column(Float, default=0.0)
    se_count = Column(Integer, default=0)
    se_area_pct = Column(Float, default=0.0)
    total_lesion_pixels = Column(Integer, default=0)

    # 4. Anatomical Landmark Localization & Biomarkers
    optic_disc_x = Column(Integer, nullable=True)
    optic_disc_y = Column(Integer, nullable=True)
    optic_disc_radius = Column(Integer, nullable=True)
    fovea_x = Column(Integer, nullable=True)
    fovea_y = Column(Integer, nullable=True)
    fovea_dist_dd = Column(Float, nullable=True)
    csme_risk = Column(String(100), nullable=True)
    eye_laterality = Column(String(50), default="Right Eye (OD)")
    microns_per_pixel = Column(Float, nullable=True)
    etdrs_json = Column(Text, nullable=True)
    uncertainty_json = Column(Text, nullable=True)

    # Retinal Quantitative Metrics (Legacy compat)
    cup_to_disc_ratio = Column(Float, nullable=True)     # Glaucoma metric (0.2 to 0.9)
    macular_integrity = Column(Float, nullable=True)     # AMD metric (0 to 100%)
    exudate_density = Column(Float, nullable=True)       # DR metric (0 to 100%)
    vessel_tortuosity = Column(Float, nullable=True)     # Hypertensive Retinopathy metric

    # Clinician notes & observations
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="scans")

    def __repr__(self):
        return f"<ScanRecord {self.id} - {self.predicted_disease} ({self.severity})>"
