"""
Patient Disease Progression & Historical Trajectory Analyzer.
Compares chronological fundus scans to evaluate disease evolution,
Cup-to-Disc Ratio shifts, exudate dynamics, and visual stability trends.
"""

from datetime import datetime
from typing import List
from app.database.models import ScanRecord, Patient

SEVERITY_ORDER = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferative / Advanced": 4
}

def analyze_patient_progression(patient: Patient, scans: List[ScanRecord]) -> dict:
    """
    Evaluates multi-visit patient records and computes clinical trajectories.
    """
    if not scans:
        return {
            "has_history": False,
            "total_scans": 0,
            "trajectory": "No Data",
            "trajectory_badge": "slate",
            "summary": "No historical fundus scans recorded for this patient yet.",
            "dates": [],
            "risk_scores": [],
            "cdr_values": [],
            "macular_values": [],
            "exudate_values": [],
            "severities": [],
            "latest_scan": None,
            "baseline_scan": None
        }

    sorted_scans = sorted(scans, key=lambda s: s.scan_date)
    total_scans = len(sorted_scans)
    baseline = sorted_scans[0]
    latest = sorted_scans[-1]

    # Time series data for interactive charts
    dates = [s.scan_date.strftime("%d %b %Y") for s in sorted_scans]
    risk_scores = [round(float(s.risk_score), 1) for s in sorted_scans]
    cdr_values = [round(float(s.cup_to_disc_ratio or 0.3), 2) for s in sorted_scans]
    macular_values = [round(float(s.macular_integrity or 95.0), 1) for s in sorted_scans]
    exudate_values = [round(float(s.exudate_density or 0.0), 1) for s in sorted_scans]
    severities = [s.severity for s in sorted_scans]

    if total_scans == 1:
        trajectory = "Baseline Established"
        trajectory_badge = "blue"
        summary = f"Initial baseline scan registered on {dates[0]} indicating {latest.severity} {latest.predicted_disease}. Subsequent follow-up scans will establish progression trendlines."
    else:
        # Compare baseline vs latest
        risk_diff = latest.risk_score - baseline.risk_score
        sev_diff = SEVERITY_ORDER.get(latest.severity, 0) - SEVERITY_ORDER.get(baseline.severity, 0)
        cdr_diff = (latest.cup_to_disc_ratio or 0.3) - (baseline.cup_to_disc_ratio or 0.3)

        if sev_diff > 0 or risk_diff > 10.0 or cdr_diff > 0.08:
            trajectory = "Deteriorating / Progression Detected"
            trajectory_badge = "rose"
            summary = (
                f"Clinical worsening detected across {total_scans} visits. "
                f"Severity escalated from {baseline.severity} to {latest.severity} "
                f"(Risk score increased by {abs(round(risk_diff, 1))} pts). "
                f"Immediate clinical reassessment and therapy escalation recommended."
            )
        elif sev_diff < 0 or risk_diff < -10.0:
            trajectory = "Improving / Positive Response"
            trajectory_badge = "emerald"
            summary = (
                f"Favorable clinical improvement observed across {total_scans} visits. "
                f"Risk score decreased from {baseline.risk_score} to {latest.risk_score} "
                f"with regression of active pathological biomarkers."
            )
        else:
            trajectory = "Stable / Quiescent"
            trajectory_badge = "amber"
            summary = (
                f"Pathology remains stable across {total_scans} visits spanning "
                f"{(latest.scan_date - baseline.scan_date).days} days. "
                f"Severity maintained at {latest.severity} with no accelerated biomarker drift."
            )

    return {
        "has_history": True,
        "total_scans": total_scans,
        "trajectory": trajectory,
        "trajectory_badge": trajectory_badge,
        "summary": summary,
        "dates": dates,
        "risk_scores": risk_scores,
        "cdr_values": cdr_values,
        "macular_values": macular_values,
        "exudate_values": exudate_values,
        "severities": severities,
        "latest_scan": latest,
        "baseline_scan": baseline,
        "scans": sorted_scans
    }
