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

    # Relationship to scans
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
    gradcam_filename = Column(String(255), nullable=True)
    pdf_report_filename = Column(String(255), nullable=True)
    
    # Clinical AI Predictions
    predicted_disease = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)  # Normal, Mild, Moderate, Severe, Proliferative/Advanced
    confidence = Column(Float, nullable=False)      # 0.0 to 1.0
    risk_score = Column(Float, nullable=False)      # 0 to 100
    
    # Retinal Quantitative Metrics
    cup_to_disc_ratio = Column(Float, nullable=True)     # Glaucoma metric (0.2 to 0.9)
    macular_integrity = Column(Float, nullable=True)     # AMD metric (0 to 100%)
    exudate_density = Column(Float, nullable=True)       # DR metric (0 to 100%)
    vessel_tortuosity = Column(Float, nullable=True)     # Hypertensive Retinopathy metric
    
    # Clinician notes & observations
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="scans")

    def __repr__(self):
        return f"<ScanRecord {self.id} - {self.predicted_disease} ({self.severity})>"
