# AI-Based Eye Disease Identification and Health Recommendation System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch_2.4-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/Theme-Light_Medical-0284c7)](#)
[![3D-Visualizer](https://img.shields.io/badge/3D-Three.js_WebGL-000000?logo=three.js)](https://threejs.org/)
[![XAI](https://img.shields.io/badge/Explainable_AI-Grad--CAM-10b981)](#)

---

## Academic Credentials & Project Metadata
- **Project Title**: AI-BASED EYE DISEASE IDENTIFICATION AND HEALTH RECOMMENDATION SYSTEM
- **Batch No**: 33
- **Hall Ticket / Candidate IDs**:
  - `23551A42B3`
  - `23551A4275`
  - `24555A4207`

---

## Abstract
Eye diseases can lead to serious vision problems if they are not detected at an early stage. This project proposes an **AI-Based Eye Disease Identification and Health Recommendation System** that uses Artificial Intelligence (AI) and Deep Learning to identify eye diseases from retinal (fundus) images. The system can detect common eye diseases, classify their severity, and use **Explainable AI (XAI)** to help users understand the prediction. 

After identifying the disease, the system provides **personalized food recommendations, nutritional guidance, eye care and lifestyle suggestions** to help improve eye health. For severe cases, it recommends **nearby eye specialists and hospitals** for immediate medical attention. The system also stores patient history, **monitors disease progression by comparing previous reports**, and generates **downloadable medical reports**. By combining disease identification, health recommendations, and patient monitoring in a single platform, the proposed system aims to support early diagnosis, improve eye care, and help users make better healthcare decisions.

---

## Key Features & Highlights

1. **Strictly Light Clinical Modern Theme**:
   - Designed exclusively in clean medical white (`#FFFFFF`, `#F8FAFC`), surgical teals, and cerulean blue accents. **No dark theme**.
2. **Interactive 3D Eyeball (Three.js)**:
   - Real-time 3D rendered anatomical eye globe. Rotate, tilt, inspect, and toggle camera focus between Cornea, Lens, Retina, Optic Disc, and Macula.
3. **Deep Learning Diagnostic Classifier**:
   - High-accuracy PyTorch transfer learning model with MobileNetV3 feature extractor.
   - Detects 6 key categories:
     - **Diabetic Retinopathy (DR)**
     - **Glaucoma**
     - **Cataract**
     - **Age-Related Macular Degeneration (AMD)**
     - **Hypertensive Retinopathy**
     - **Healthy Normal Retina**
4. **Explainable AI (XAI) with Grad-CAM**:
   - Hooks into neural feature activation layers to generate high-resolution Jet/Turbo heatmaps that pinpoint pathognomonic lesions (microaneurysms, hard exudates, optic nerve cupping, drusen).
   - Interactive multi-view comparator (Original Fundus, CLAHE Green-Channel Enhancement, Grad-CAM Heatmap, Overlay).
5. **Personalized Nutrition & Daily Eye Care Plan**:
   - Prescribes essential retinal micronutrients (Lutein, Zeaxanthin, Omega-3, AREDS-2, Alpha-Lipoic Acid, Zinc).
   - Structured daily meal plans (Breakfast, Lunch, Snack, Dinner).
   - Strict "Foods to Avoid" with clinical reasoning.
   - Ergonomic screen-use routines (20-20-20 rule, IOP management, lighting).
6. **Patient History & Longitudinal Progression Monitoring**:
   - SQLite database with multi-scan patient timelines.
   - Interactive Chart.js analytics: Risk Score drift, Cup-to-Disc Ratio (CDR) trendlines.
   - Trajectory classifier: *Improving*, *Stable*, or *Deteriorating*.
7. **Eye Specialist & Hospital Directory**:
   - Curated directory of tertiary eye institutes and subspecialists for severe case escalation.
   - Distance indicators, contact details, 24/7 trauma hotlines, and Google Maps directions.
8. **Downloadable Clinical PDF Reports**:
   - Generated using ReportLab with clinical tables, embedded retinal and Grad-CAM scans, patient demographics, dietary prescriptions, and doctor sign-off blocks.

---

## Project Architecture

```
eye/
├── app/
│   ├── database/
│   │   ├── db.py               # SQLite connection, session & demo seeder
│   │   └── models.py           # SQLAlchemy Patient & ScanRecord models
│   ├── ml/
│   │   ├── gradcam.py          # PyTorch Grad-CAM XAI implementation
│   │   ├── model.py            # Deep learning model & clinical inference engine
│   │   ├── preprocessor.py     # Retinal circular crop & CLAHE enhancement
│   │   └── presets.py          # Synthetic fundus preset image generator
│   ├── services/
│   │   ├── hospitals.py        # Tertiary eye hospital & specialist locator
│   │   ├── pdf_report.py       # ReportLab medical PDF report generator
│   │   ├── progression.py      # Multi-visit trajectory & trend analyzer
│   │   └── recommendations.py  # Personalized nutrition & eye-care engine
│   ├── static/
│   │   ├── css/styles.css      # Light theme, laser scan animations, 3D tilt
│   │   ├── js/eye_3d.js        # Three.js 3D eyeball visualization
│   │   ├── images/presets/     # Sample fundus images for 1-click testing
│   │   ├── uploads/            # Processed scan uploads & Grad-CAM outputs
│   │   └── reports/            # Generated PDF clinical reports
│   ├── templates/
│   │   ├── base.html           # Master layout with Batch 33 accreditation
│   │   ├── index.html          # Home landing page with 3D eye viewer
│   │   ├── diagnose.html       # Fundus scan upload studio & 1-click presets
│   │   ├── result.html         # Diagnostic report, XAI slider & nutrition plan
│   │   ├── patients.html       # Patient directory & record management
│   │   ├── progression.html    # Historical progression tracker with charts
│   │   ├── hospitals.html      # Eye specialist & hospital referral network
│   │   └── about.html          # Project documentation, abstract & team
│   └── main.py                 # FastAPI application routes & lifespan
├── tests/
│   └── test_app.py             # Automated unit & integration tests
├── Dockerfile                  # Production container definition
├── requirements.txt            # Python dependencies
├── run.py                      # Application launcher script
└── README.md                   # Full documentation
```

---

## Installation & Running Locally

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git

### 2. Clone and Setup Environment
```bash
# Navigate to directory
cd eye

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Server
```bash
python run.py
```
Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## 1-Click Instant Demonstration
On the AI Diagnosis page (`/diagnose`), simply click any of the preset cards (e.g., **Diabetic Retinopathy**, **Glaucoma**, **Cataract**, **AMD**, or **Healthy Normal**) to immediately trigger deep learning classification, Grad-CAM heatmap generation, and personalized health recommendations without having to upload external files!

---

## Running Automated Tests
```bash
pytest tests/ -v
```

---

## Docker Deployment (Market Ready)
```bash
# Build the Docker image
docker build -t eye-disease-ai .

# Run the container
docker run -p 8000:8000 eye-disease-ai
```
Visit `http://localhost:8000` to access the live application.
