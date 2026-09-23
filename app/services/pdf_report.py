"""
Downloadable Medical PDF Report Generator using ReportLab.
Produces comprehensive, hospital-grade diagnostic reports incorporating
patient demographics, fundus photography, Grad-CAM XAI heatmaps,
clinical biomarker tables, and tailored nutritional prescriptions.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

REPORTS_DIR = "app/static/reports"

def generate_medical_report_pdf(
    patient_info: dict,
    scan_info: dict,
    original_img_path: str,
    gradcam_img_path: str,
    recommendations: dict,
    output_filename: str
) -> str:
    """
    Builds and writes a professional clinical PDF report.
    Returns the absolute path of the generated PDF file.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORTS_DIR, output_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom light theme palette styles
    primary_color = colors.HexColor("#0284C7")    # Cerulean / Medical Blue
    teal_color = colors.HexColor("#0D9488")       # Surgical Teal
    dark_slate = colors.HexColor("#0F172A")       # Deep text
    light_bg = colors.HexColor("#F8FAFC")         # Clean medical background
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=primary_color,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        alignment=1
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=teal_color,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=dark_slate
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=dark_slate
    )

    alert_style = ParagraphStyle(
        'Alert',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#B91C1C")
    )

    elements = []

    # 1. Header Banner
    header_html = "AI-BASED EYE DISEASE IDENTIFICATION & HEALTH RECOMMENDATION SYSTEM"
    elements.append(Paragraph(header_html, title_style))
    elements.append(Paragraph("BATCH NO: 33 &nbsp;|&nbsp; 23551A42B3 &nbsp;•&nbsp; 23551A4275 &nbsp;•&nbsp; 24555A4207", subtitle_style))
    elements.append(Paragraph("CLINICAL OPHTHALMOLOGY & EXPLAINABLE AI DIAGNOSTIC REPORT", ParagraphStyle('ReportType', parent=subtitle_style, textColor=primary_color, fontSize=8)))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=10))

    # 2. Patient Demographics & Exam Info Table
    patient_table_data = [
        [
            Paragraph("<b>Patient Name:</b> " + str(patient_info.get("name", "N/A")), body_style),
            Paragraph("<b>Patient ID:</b> " + str(patient_info.get("patient_id", "N/A")), body_style),
            Paragraph("<b>Exam Date:</b> " + scan_info.get("scan_date", datetime.utcnow().strftime("%d %b %Y")), body_style)
        ],
        [
            Paragraph("<b>Age / Gender:</b> " + f"{patient_info.get('age', '--')} yrs / {patient_info.get('gender', '--')}", body_style),
            Paragraph("<b>Contact:</b> " + str(patient_info.get("contact", "--")), body_style),
            Paragraph("<b>Medical History:</b> " + str(patient_info.get("medical_history", "None recorded")), body_style)
        ]
    ]

    p_table = Table(patient_table_data, colWidths=[2.3*inch, 2.2*inch, 2.8*inch])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.8, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(p_table)
    elements.append(Spacer(1, 12))

    # 3. Primary AI Diagnostic Assessment
    elements.append(Paragraph("1. ARTIFICIAL INTELLIGENCE DIAGNOSTIC SUMMARY", section_header))
    
    disease_name = scan_info.get("predicted_disease", "Normal")
    severity = scan_info.get("severity", "Normal")
    confidence = scan_info.get("confidence", 0.95) * 100
    risk_score = scan_info.get("risk_score", 10.0)

    diag_data = [
        [
            Paragraph("<b>Detected Condition</b>", body_bold),
            Paragraph("<b>Severity Staging</b>", body_bold),
            Paragraph("<b>AI Confidence</b>", body_bold),
            Paragraph("<b>Overall Risk Index</b>", body_bold)
        ],
        [
            Paragraph(f"<font size=11 color='#0284C7'><b>{disease_name}</b></font>", body_style),
            Paragraph(f"<b>{severity}</b>", body_style),
            Paragraph(f"{confidence:.1f}%", body_style),
            Paragraph(f"{risk_score:.1f} / 100", body_style)
        ]
    ]
    diag_table = Table(diag_data, colWidths=[2.5*inch, 1.8*inch, 1.5*inch, 1.5*inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E0F2FE")),
        ('BOX', (0, 0), (-1, -1), 1, primary_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 10))

    # 4. Clinical Retinal Biometric Metrics
    biomarkers = scan_info.get("biomarkers", {})
    bio_data = [
        [
            Paragraph(f"<b>Cup-to-Disc Ratio (CDR):</b> {biomarkers.get('cup_to_disc_ratio', scan_info.get('cup_to_disc_ratio', 0.32))}", body_style),
            Paragraph(f"<b>Macular Integrity:</b> {biomarkers.get('macular_integrity', scan_info.get('macular_integrity', 96.0))}%", body_style),
            Paragraph(f"<b>Lesion Density:</b> {biomarkers.get('exudate_density', scan_info.get('exudate_density', 0.0))}%", body_style),
            Paragraph(f"<b>Vessel Tortuosity:</b> {biomarkers.get('vessel_tortuosity', scan_info.get('vessel_tortuosity', 1.05))}", body_style)
        ]
    ]
    bio_table = Table(bio_data, colWidths=[1.85*inch, 1.85*inch, 1.85*inch, 1.75*inch])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(bio_table)
    elements.append(Spacer(1, 12))

    # 5. Side-by-Side Fundus & Grad-CAM Visuals
    elements.append(Paragraph("2. EXPLAINABLE AI (XAI) GRAD-CAM LESION LOCALIZATION", section_header))
    
    img_cells = []
    if os.path.exists(original_img_path):
        img_cells.append([
            RLImage(original_img_path, width=2.4*inch, height=2.4*inch),
            Paragraph("<div align='center'><b>Original Retinal Fundus</b><br/><font size=7 color='#64748B'>Full vascular tree & optic disc</font></div>", body_style)
        ])
    else:
        img_cells.append([Paragraph("Original Image Not Found", body_style), Paragraph("", body_style)])

    if os.path.exists(gradcam_img_path):
        img_cells.append([
            RLImage(gradcam_img_path, width=2.4*inch, height=2.4*inch),
            Paragraph("<div align='center'><b>Grad-CAM Explainability Heatmap</b><br/><font size=7 color='#64748B'>Warm regions (Red/Yellow) highlight focal lesions</font></div>", body_style)
        ])
    else:
        img_cells.append([Paragraph("Heatmap Image Not Found", body_style), Paragraph("", body_style)])

    visual_table_data = [
        [img_cells[0][0], img_cells[1][0]],
        [img_cells[0][1], img_cells[1][1]]
    ]
    visual_table = Table(visual_table_data, colWidths=[3.6*inch, 3.6*inch])
    visual_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(visual_table)
    elements.append(Spacer(1, 12))

    # 6. Personalized Nutritional Prescriptions & Lifestyle Directives
    elements.append(Paragraph("3. PERSONALIZED NUTRITIONAL & EYE CARE PRESCRIPTION", section_header))
    
    nutrients = recommendations.get("nutrients", [])[:3]
    nutr_texts = []
    for n in nutrients:
        nutr_texts.append(f"• <b>{n['name']}</b>: {n['benefit']} (<i>Food sources: {n['sources']}</i>)")

    foods_avoid = recommendations.get("foods_to_avoid", [])[:2]
    avoid_texts = []
    for f in foods_avoid:
        avoid_texts.append(f"• <b>{f['item']}</b>: {f['reason']}")

    lifestyle_habits = recommendations.get("eye_care_routine", [])[:2]
    life_texts = [f"• {h}" for h in lifestyle_habits]

    rec_data = [
        [Paragraph("<b>Target Nutrients:</b><br/>" + "<br/>".join(nutr_texts), body_style)],
        [Paragraph("<b>Foods to Avoid:</b><br/>" + "<br/>".join(avoid_texts), body_style)],
        [Paragraph("<b>Daily Eye Care Habits (20-20-20 Rule):</b><br/>" + "<br/>".join(life_texts), body_style)]
    ]
    rec_table = Table(rec_data, colWidths=[7.3*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 10))

    # 7. Urgency & Specialist Referral Notice (if severe or moderate)
    if recommendations.get("hospital_recommendation", False):
        elements.append(Paragraph("<b>CLINICAL ACTION REQUIRED:</b> " + recommendations.get("urgency", "Referral Advised"), alert_style))
        elements.append(Paragraph("<i>" + recommendations.get("follow_up", "Please visit an ophthalmologist promptly.") + "</i>", body_style))
        elements.append(Spacer(1, 10))

    # 8. Doctor Sign-off & Academic Disclaimer Block
    sign_data = [
        [
            Paragraph("<b>System Certification:</b><br/><font size=7 color='#64748B'>AI-assisted screening tool for clinical decision support. Final diagnostic confirmation must be validated by a registered ophthalmologist.</font>", body_style),
            Paragraph("<b>Consulting Ophthalmologist:</b><br/><br/>___________________________________<br/><font size=7 color='#64748B'>Signature & Stamp</font>", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.5*inch, 2.8*inch])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(KeepTogether(sign_table))

    # Build PDF document
    doc.build(elements)
    return pdf_path
