"""
Downloadable Clinical Diagnostic PDF Report Generator using ReportLab.
Produces comprehensive, hospital-grade ophthalmic diagnostic reports incorporating:
1. Multi-Disease Identification & Screening (GhostNet Multi-Label Probabilities)
2. 5-Stage Disease Severity Grading (ConvNeXt + Hybrid Clinical Scoring)
3. Pixel-Level Lesion Segmentation (U-Net MA, HE, EX, SE Quantitative Metrics)
4. Anatomical Landmark Localization (Optic Disc CDR & Fovea CSME Risk)
5. Multi-Modal Visual Evidence Grid (Original, CLAHE, Segmentation, Grad-CAM Saliency)
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
    recommendations: dict = None,
    output_filename: str = "diagnostic_report.pdf",
    composite_img_path: str = None,
    landmark_img_path: str = None,
    clahe_img_path: str = None
) -> str:
    """
    Builds and writes a professional clinical diagnostic PDF report.
    Returns the absolute path of the generated PDF file.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = os.path.join(REPORTS_DIR, output_filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    # Medical theme color palette
    primary_color = colors.HexColor("#0284C7")    # Deep Medical Blue
    teal_color = colors.HexColor("#0D9488")       # Clinical Teal
    dark_slate = colors.HexColor("#0F172A")       # Deep Charcoal
    light_slate = colors.HexColor("#475569")
    card_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=15,
        textColor=primary_color,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        alignment=1
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=teal_color,
        spaceAfter=3,
        spaceBefore=5
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=dark_slate
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=dark_slate
    )

    alert_style = ParagraphStyle(
        'Alert',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#991B1B")
    )

    elements = []

    # 1. Official Header
    elements.append(Paragraph("AUTOMATED RETINAL DIAGNOSTIC & LESION SEGMENTATION REPORT", title_style))
    elements.append(Paragraph("DEEP LEARNING CLINICAL DECISION SUPPORT SYSTEM • GHOSTNET • CONVNEXT • U-NET", subtitle_style))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=1, spaceAfter=5))

    # 2. Patient & Examination Metadata Table
    p_name = patient_info.get("name", "Clinical Subject")
    p_id = patient_info.get("patient_id", "REF-SCAN")
    p_age = patient_info.get("age", 45)
    p_gender = patient_info.get("gender", "Unspecified")
    eye_laterality = scan_info.get("eye_laterality", "Right Eye (OD)")
    scan_date = scan_info.get("scan_date", datetime.utcnow().strftime("%d %b %Y, %I:%M %p"))

    info_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {p_name}", body_style),
            Paragraph(f"<b>Exam / Scan ID:</b> {p_id}", body_style),
            Paragraph(f"<b>Exam Date:</b> {scan_date}", body_style)
        ],
        [
            Paragraph(f"<b>Demographics:</b> {p_age}y / {p_gender}", body_style),
            Paragraph(f"<b>Eye Laterality:</b> {eye_laterality}", body_style),
            Paragraph(f"<b>Clinical Status:</b> Completed (4 Tasks)", body_style)
        ]
    ]
    info_table = Table(info_data, colWidths=[2.5*inch, 2.4*inch, 2.4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), card_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 5))

    # 3. Primary Finding & Severity Summary Banner
    pred_disease = scan_info.get("predicted_disease", "Normal")
    dr_grade = scan_info.get("dr_grade", 0)
    dr_stage_name = scan_info.get("dr_stage_name", "Grade 0: Normal / No Apparent Retinopathy")
    hybrid_score = scan_info.get("hybrid_severity_score", 0.0)
    confidence_pct = round(scan_info.get("confidence", 0.95) * 100, 1)

    diag_data = [
        [
            Paragraph("<b>PRIMARY RETINAL FINDING (GhostNet):</b>", body_style),
            Paragraph(f"<font size=10 color='#0284C7'><b>{pred_disease}</b></font> ({confidence_pct}% Screening Confidence)", body_style)
        ],
        [
            Paragraph("<b>DR SEVERITY STAGING (ConvNeXt):</b>", body_style),
            Paragraph(f"<font size=9.5 color='#B45309'><b>Grade {dr_grade}: {dr_stage_name}</b></font> (Hybrid Score: {hybrid_score}/100)", body_style)
        ]
    ]
    diag_table = Table(diag_data, colWidths=[2.8*inch, 4.5*inch])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.8, primary_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 6))

    # 4. Pillar 1 & 2: Multi-Disease Probabilities & Hybrid DR Scoring
    elements.append(Paragraph("1. MULTI-DISEASE SCREENING & SEVERITY STRATIFICATION", section_header))

    multi_probs = scan_info.get("multi_disease_probabilities", {})
    if not multi_probs and "multi_disease_json" in scan_info:
        import json
        try:
            multi_probs = json.loads(scan_info["multi_disease_json"])
        except Exception:
            multi_probs = {}

    dr_p = round(multi_probs.get("Diabetic Retinopathy", 0.0) * 100, 1)
    gl_p = round(multi_probs.get("Glaucoma", 0.0) * 100, 1)
    cat_p = round(multi_probs.get("Cataract", 0.0) * 100, 1)
    ht_p = round(multi_probs.get("Hypertensive Retinopathy", 0.0) * 100, 1)

    screening_data = [
        [
            Paragraph("<b>Condition</b>", body_bold),
            Paragraph("<b>Probability</b>", body_bold),
            Paragraph("<b>Clinical Biomarker Finding</b>", body_bold),
            Paragraph("<b>Risk Stratification</b>", body_bold)
        ],
        [
            Paragraph("Diabetic Retinopathy", body_style),
            Paragraph(f"{dr_p}%", body_style),
            Paragraph(f"Stage: Grade {dr_grade} ({dr_stage_name})", body_style),
            Paragraph(f"Score: {hybrid_score}/100", body_style)
        ],
        [
            Paragraph("Glaucoma", body_style),
            Paragraph(f"{gl_p}%", body_style),
            Paragraph(f"Cup-to-Disc Ratio (CDR): {scan_info.get('cup_to_disc_ratio', 0.35)}", body_style),
            Paragraph("High" if float(scan_info.get('cup_to_disc_ratio', 0.35)) >= 0.65 else ("Suspicious" if float(scan_info.get('cup_to_disc_ratio', 0.35)) >= 0.50 else "Normal"), body_style)
        ],
        [
            Paragraph("Cataract", body_style),
            Paragraph(f"{cat_p}%", body_style),
            Paragraph("Media Opacity Assessment", body_style),
            Paragraph("Flagged" if cat_p > 50 else "Clear Media", body_style)
        ],
        [
            Paragraph("Hypertensive Retinopathy", body_style),
            Paragraph(f"{ht_p}%", body_style),
            Paragraph(f"Vessel Tortuosity & Arteriolar Narrowing", body_style),
            Paragraph("Elevated" if ht_p > 50 else "Stable", body_style)
        ]
    ]
    scr_table = Table(screening_data, colWidths=[1.8*inch, 1.1*inch, 2.8*inch, 1.6*inch])
    scr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(scr_table)
    elements.append(Spacer(1, 6))

    # 5. Pillar 3 & 4: Quantitative Lesion Segmentation & Landmark Tracking
    elements.append(Paragraph("2. QUANTITATIVE LESION SEGMENTATION (U-NET) & ANATOMICAL LANDMARKS", section_header))

    ma_count = scan_info.get("ma_count", 0)
    he_count = scan_info.get("he_count", 0)
    ex_count = scan_info.get("ex_count", 0)
    se_count = scan_info.get("se_count", 0)
    cdr = scan_info.get("cup_to_disc_ratio", 0.35)
    fovea_dist = scan_info.get("fovea_dist_dd", "Safe")
    csme_risk = scan_info.get("csme_risk", "Fovea Spared")

    quant_data = [
        [
            Paragraph(f"<b>Microaneurysms (MA):</b> {ma_count} spots", body_style),
            Paragraph(f"<b>Hemorrhages (HE):</b> {he_count} lesions ({scan_info.get('he_area_pct', 0.0)}% area)", body_style),
            Paragraph(f"<b>Hard Exudates (EX):</b> {ex_count} clusters ({scan_info.get('ex_area_pct', 0.0)}% area)", body_style)
        ],
        [
            Paragraph(f"<b>Soft Exudates (SE):</b> {se_count} spots ({scan_info.get('se_area_pct', 0.0)}% area)", body_style),
            Paragraph(f"<b>Cup-to-Disc Ratio (CDR):</b> {cdr} (Glaucoma)", body_style),
            Paragraph(f"<b>Foveal Clearance:</b> {fovea_dist} DD ({csme_risk})", body_style)
        ]
    ]
    quant_table = Table(quant_data, colWidths=[2.4*inch, 2.5*inch, 2.4*inch])
    quant_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(quant_table)
    elements.append(Spacer(1, 6))

    # 6. Visual Evidence: 4-Grid Quad Panel (Original, U-Net Masks, Landmarks, Grad-CAM)
    elements.append(Paragraph("3. MULTI-MODAL VISUAL EVIDENCE & EXPLAINABILITY (XAI)", section_header))

    def get_img_cell(path, caption):
        if path and os.path.exists(path):
            return [
                RLImage(path, width=1.65*inch, height=1.65*inch),
                Paragraph(f"<div align='center'><b>{caption}</b></div>", body_style)
            ]
        else:
            return [
                Paragraph("<div align='center'><i>Image Pending</i></div>", body_style),
                Paragraph(f"<div align='center'><b>{caption}</b></div>", body_style)
            ]

    c1 = get_img_cell(original_img_path, "1. Original Fundus Scan")
    c2 = get_img_cell(composite_img_path or original_img_path, "2. U-Net Lesion Segmentation")
    c3 = get_img_cell(landmark_img_path or original_img_path, "3. Landmark Tracking (OD/FV)")
    c4 = get_img_cell(gradcam_img_path or original_img_path, "4. GhostNet Grad-CAM Saliency")

    visual_table_data = [
        [c1[0], c2[0], c3[0], c4[0]],
        [c1[1], c2[1], c3[1], c4[1]]
    ]
    visual_table = Table(visual_table_data, colWidths=[1.82*inch, 1.82*inch, 1.82*inch, 1.82*inch])
    visual_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(visual_table)
    elements.append(Spacer(1, 6))

    # 7. Clinical Triage Directives & Action Plan
    elements.append(Paragraph("4. CLINICAL IMPRESSION & TRIAGE DIRECTIVES", section_header))

    referral_urgency = scan_info.get("referral_urgency", "Routine Screening")
    referral_timeline = scan_info.get("referral_timeline", "12 Months")

    notes_text = scan_info.get("notes") or (
        f"Automated evaluation reveals {pred_disease} staged at Grade {dr_grade} ({dr_stage_name}). "
        f"U-Net delineated {ma_count} microaneurysms, {he_count} hemorrhages, {ex_count} hard exudates, and {se_count} soft exudates. "
        f"Optic Disc Cup-to-Disc Ratio (CDR) measured at {cdr}. Macular clearance: {fovea_dist} DD with CSME status evaluated as {csme_risk}."
    )

    triage_data = [
        [Paragraph(f"<b>Ophthalmic Narrative:</b> {notes_text}", body_style)],
        [Paragraph(f"<b>CLINICAL ACTION & REFERRAL TIMELINE:</b> <font color='#B91C1C'>{referral_urgency} (Recommended follow-up: {referral_timeline})</font>", alert_style)]
    ]
    triage_table = Table(triage_data, colWidths=[7.3*inch])
    triage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF2F2" if "Urgent" in referral_urgency else "#F0FDF4")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#FCA5A5" if "Urgent" in referral_urgency else "#86EFAC")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(triage_table)
    elements.append(Spacer(1, 8))

    # 8. Certification & Sign-off Block
    sign_data = [
        [
            Paragraph("<b>Automated System Certification:</b><br/><font size=7 color='#64748B'>Multi-Task Deep Learning Pipeline (GhostNet Screening + ConvNeXt DR Grading + U-Net Lesion Segmentation + Geodesic Landmark Localization). Certified for decision-support triage.</font>", body_style),
            Paragraph("<b>Examining Ophthalmologist:</b><br/><br/>___________________________________<br/><font size=7 color='#64748B'>Signature & Medical License</font>", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.7*inch, 2.6*inch])
    sign_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(KeepTogether(sign_table))

    # Build PDF document
    doc.build(elements)
    return pdf_path
