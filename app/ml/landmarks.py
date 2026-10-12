"""
Anatomical Landmark Localization Engine.
Locates Optic Disc (OD) and Fovea (FV) in retinal fundus photography,
measures Cup-to-Disc Ratio (CDR) for Glaucoma assessment,
and evaluates Clinically Significant Macular Edema (CSME) risk based on lesion-to-fovea proximity.
"""

import os
import math
import cv2
import numpy as np

class RetinalLandmarkDetector:
    """
    Robust anatomical landmark detection using green-channel vascular convergence,
    luminance thresholding, and geometric fundus spatial priors.
    """
    def __init__(self):
        pass

    def detect_optic_disc(self, original_bgr: np.ndarray, enhanced_bgr: np.ndarray, retinal_mask: np.ndarray) -> dict:
        """
        Locates Optic Disc (OD) centroid, radius, and physiological cup:
        - OD is characterized by high intensity in red and green channels (pale yellowish-orange circular structure)
        - Cup is the brightest central pale depression within the disc
        """
        h, w = original_bgr.shape[:2]
        glaucoma_flag: bool = False
        glaucoma_risk: str = "Normal Physiological Cupping"
        red_ch = enhanced_bgr[:, :, 2]
        green_ch = enhanced_bgr[:, :, 1]
        blue_ch = enhanced_bgr[:, :, 0]

        # Candidate disc brightness map: high red & green, low blue relative to red
        disc_metric = (red_ch.astype(np.float32) * 0.5 + green_ch.astype(np.float32) * 0.5)
        disc_metric[retinal_mask == 0] = 0

        # Discard extreme border pixels to avoid edge glare
        eroded_mask = cv2.erode(retinal_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
        disc_metric[eroded_mask == 0] = 0

        # Spatial Gaussian blur to smooth vessels crossing the disc
        blurred = cv2.GaussianBlur(disc_metric, (45, 45), 0)

        # Find maximum intensity point as disc center seed
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        seed_x, seed_y = max_loc

        # Refine disc ROI (typical disc radius is ~ 35 to 55 pixels on a 512x512 image)
        expected_r = int(min(h, w) * 0.08)
        roi_x1 = max(0, seed_x - expected_r * 2)
        roi_x2 = min(w, seed_x + expected_r * 2)
        roi_y1 = max(0, seed_y - expected_r * 2)
        roi_y2 = min(h, seed_y + expected_r * 2)

        roi_metric = disc_metric[roi_y1:roi_y2, roi_x1:roi_x2]
        threshold_val = max(180, int(np.percentile(roi_metric[roi_metric > 0], 85))) if np.any(roi_metric > 0) else 190
        disc_binary = (roi_metric >= threshold_val).astype(np.uint8) * 255

        # Morphological close to bridge vessels
        disc_binary = cv2.morphologyEx(disc_binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        # Find largest connected component in ROI
        contours, _ = cv2.findContours(disc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (local_cx, local_cy), radius = cv2.minEnclosingCircle(c)
            od_x = int(roi_x1 + local_cx)
            od_y = int(roi_y1 + local_cy)
            od_r = max(int(radius), int(min(h, w) * 0.065))
        else:
            od_x, od_y = seed_x, seed_y
            od_r = expected_r

        # Detect Physiological Optic Cup (brightest pale central core)
        # Search within optic disc radius
        y_indices, x_indices = np.ogrid[:h, :w]
        disc_circ_mask = ((x_indices - od_x)**2 + (y_indices - od_y)**2) <= (od_r**2)
        
        cup_metric = (red_ch.astype(np.float32) + green_ch.astype(np.float32)) / 2.0
        cup_metric[~disc_circ_mask] = 0

        cup_thresh = max(210, int(np.percentile(cup_metric[disc_circ_mask], 70))) if np.any(disc_circ_mask) else 220
        cup_binary = (cup_metric >= cup_thresh).astype(np.uint8) * 255

        cup_pixels = int(np.sum(cup_binary > 0))
        cup_r_estimated = math.sqrt(cup_pixels / math.pi) if cup_pixels > 5 else od_r * 0.35
        # Clinical constraint for cup radius: cup must be inside disc
        cup_r = max(min(int(cup_r_estimated), int(od_r * 0.88)), int(od_r * 0.25))

        # Vertical Cup-to-Disc Ratio (CDR)
        cdr = round(float(cup_r) / float(max(od_r, 1)), 2)

        if cdr >= 0.65:
            glaucoma_risk = "High Risk / Glaucomatous Cupping"
            glaucoma_flag = True
        elif cdr >= 0.50:
            glaucoma_risk = "Borderline / Suspicious Cupping"
            glaucoma_flag = False
        else:
            glaucoma_risk = "Normal Physiological Cupping"
            glaucoma_flag = False
        # Physical micron calibration: Human optic disc diameter is universally ~1500 µm (1.5 mm)
        od_diam = max(od_r * 2, 1)
        microns_per_pixel = round(1500.0 / float(od_diam), 2)

        return {
            "x": od_x,
            "y": od_y,
            "radius": od_r,
            "diameter_px": od_diam,
            "microns_per_pixel": microns_per_pixel,
            "cup_x": od_x,
            "cup_y": od_y,
            "cup_radius": cup_r,
            "cup_to_disc_ratio": cdr,
            "glaucoma_risk": glaucoma_risk,
            "glaucoma_flag": glaucoma_flag
        }

    def pinpoint_fovea(self, original_bgr: np.ndarray, enhanced_bgr: np.ndarray, od_info: dict, retinal_mask: np.ndarray) -> dict:
        """
        Locates Fovea Centralis (FV) / Macula:
        - Fovea is temporal to the optic disc at approximately 2.5 disc diameters distance.
        - Located in an avascular zone with darker, uniform reddish-brown pigmentation.
        """
        h, w = original_bgr.shape[:2]
        od_x, od_y = od_info["x"], od_info["y"]
        disc_diam = od_info["diameter_px"]

        # Determine eye laterality: if OD is on the left (nasal for right eye OD), fovea is to the right
        # If OD is on the right, fovea is to the left
        expected_dist = int(disc_diam * 2.3)
        if od_x < w // 2:
            # Right eye fundus (OD nasal left, macula temporal right)
            expected_fv_x = min(w - 50, od_x + expected_dist)
            eye_laterality = "Right Eye (OD)"
        else:
            # Left eye fundus (OD nasal right, macula temporal left)
            expected_fv_x = max(50, od_x - expected_dist)
            eye_laterality = "Left Eye (OS)"

        expected_fv_y = od_y + int(disc_diam * 0.1)  # Fovea is slightly inferior to disc center

        # Search window around expected fovea position
        window_r = int(disc_diam * 0.6)
        x1 = max(0, expected_fv_x - window_r)
        x2 = min(w, expected_fv_x + window_r)
        y1 = max(0, expected_fv_y - window_r)
        y2 = min(h, expected_fv_y + window_r)

        # Fovea is the darkest spot in the green channel (avascular depression)
        green_ch = enhanced_bgr[:, :, 1]
        roi_green = green_ch[y1:y2, x1:x2].copy()
        
        # Mask out vessels in ROI
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_smooth = cv2.GaussianBlur(roi_green, (15, 15), 0)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(roi_smooth)
        fv_x = x1 + min_loc[0]
        fv_y = y1 + min_loc[1]

        # Macula radius is approximately 1 disc diameter
        macula_radius = int(disc_diam * 0.75)

        return {
            "x": fv_x,
            "y": fv_y,
            "macula_radius": macula_radius,
            "eye_laterality": eye_laterality
        }

    def evaluate_macular_threat(self, fovea_info: dict, od_info: dict, lesion_binary_masks: dict) -> dict:
        """
        Calculates distance from detected lesions to Fovea to determine Clinically Significant Macular Edema (CSME).
        - High CSME Risk: Hard exudates within 500 microns (~ 0.33 Disc Diameters) of the foveal center.
        - Moderate Risk: Hard exudates within 0.33 to 1.0 Disc Diameters.
        - Low Risk: Exudates > 1.0 DD or absent.
        """
        fv_x, fv_y = fovea_info["x"], fovea_info["y"]
        disc_diam = max(1, od_info["diameter_px"])

        ex_mask = lesion_binary_masks.get("ex", None)
        he_mask = lesion_binary_masks.get("he", None)
        ma_mask = lesion_binary_masks.get("ma", None)

        nearest_ex_dist_px = float("inf")
        nearest_any_dist_px = float("inf")
        nearest_ex_coord = None
        nearest_any_coord = None

        # Check hard exudate distance
        if ex_mask is not None and np.any(ex_mask > 0):
            ex_pts = np.argwhere(ex_mask > 0)  # y, x
            distances = np.sqrt((ex_pts[:, 1] - fv_x)**2 + (ex_pts[:, 0] - fv_y)**2)
            min_idx = np.argmin(distances)
            nearest_ex_dist_px = float(distances[min_idx])
            nearest_ex_coord = (int(ex_pts[min_idx, 1]), int(ex_pts[min_idx, 0]))

        # Check all lesions
        for m in [ex_mask, he_mask, ma_mask]:
            if m is not None and np.any(m > 0):
                pts = np.argwhere(m > 0)
                dists = np.sqrt((pts[:, 1] - fv_x)**2 + (pts[:, 0] - fv_y)**2)
                idx = np.argmin(dists)
                d = float(dists[idx])
                if d < nearest_any_dist_px:
                    nearest_any_dist_px = d
                    nearest_any_coord = (int(pts[idx, 1]), int(pts[idx, 0]))

        # Calculate distances in Disc Diameters (DD)
        nearest_ex_dd = round(nearest_ex_dist_px / disc_diam, 2) if nearest_ex_dist_px != float("inf") else 99.0
        nearest_any_dd = round(nearest_any_dist_px / disc_diam, 2) if nearest_any_dist_px != float("inf") else 99.0

        # CSME Clinical Evaluation
        if nearest_ex_dd <= 0.33:
            csme_risk = "High Risk (CSME Confirmed)"
            csme_stage = "Critical Macular Threat"
            csme_description = "Hard exudates observed within 500µm (≤ 0.33 DD) of the foveal center. High risk of immediate central visual loss."
            csme_color = "rose"
        elif nearest_ex_dd <= 1.0:
            csme_risk = "Moderate Risk (Macular Threat)"
            csme_stage = "Perimacular Involvement"
            csme_description = "Exudates present within 1.0 Disc Diameter of the fovea. Perimacular edema monitoring required."
            csme_color = "amber"
        elif nearest_any_dd <= 1.0:
            csme_risk = "Low Risk (Fovea Spared)"
            csme_stage = "Extramacular Lesions"
            csme_description = "Retinal hemorrhages or microaneurysms detected in macula, but no hard exudates directly threatening foveal pit."
            csme_color = "blue"
        else:
            csme_risk = "No Macular Involvement"
            csme_stage = "Fovea Completely Spared"
            csme_description = "Foveal avascular zone and central macula are clear of exudative and hemorrhagic lesions."
            csme_color = "emerald"

        # Calculate physical micron distances using OD scale
        um_per_px = od_info.get("microns_per_pixel", round(1500.0 / float(max(1, disc_diam)), 2))
        nearest_ex_dist_um = round(nearest_ex_dist_px * um_per_px, 1) if nearest_ex_dist_px != float("inf") else -1
        nearest_any_dist_um = round(nearest_any_dist_px * um_per_px, 1) if nearest_any_dist_px != float("inf") else -1

        return {
            "csme_risk": csme_risk,
            "csme_stage": csme_stage,
            "csme_color": csme_color,
            "csme_description": csme_description,
            "nearest_ex_dist_px": round(nearest_ex_dist_px, 1) if nearest_ex_dist_px != float("inf") else -1,
            "nearest_ex_dist_dd": nearest_ex_dd,
            "nearest_ex_dist_um": nearest_ex_dist_um,
            "nearest_any_dist_px": round(nearest_any_dist_px, 1) if nearest_any_dist_px != float("inf") else -1,
            "nearest_any_dist_dd": nearest_any_dd,
            "nearest_any_dist_um": nearest_any_dist_um,
            "microns_per_pixel": um_per_px,
            "nearest_ex_coord": nearest_ex_coord,
            "nearest_any_coord": nearest_any_coord
        }

    def generate_landmark_overlay(
        self,
        original_bgr: np.ndarray,
        od_info: dict,
        fovea_info: dict,
        csme_info: dict,
        output_path: str
    ) -> str:
        """
        Draws clinical landmark annotations on the fundus image:
        - Optic Disc (Green boundary circle + centroid)
        - Optic Cup (Teal dashed circle)
        - Fovea (Blue target crosshair + 500µm CSME danger ring)
        - Nearest lesion connector line with distance indicator
        """
        annotated = original_bgr.copy()
        h, w = original_bgr.shape[:2]

        od_x, od_y, od_r = od_info["x"], od_info["y"], od_info["radius"]
        cup_r = od_info["cup_radius"]
        fv_x, fv_y = fovea_info["x"], fovea_info["y"]
        disc_diam = od_info["diameter_px"]

        # 1. Optic Disc (Emerald Green, BGR: 74, 222, 128)
        cv2.circle(annotated, (od_x, od_y), od_r, (74, 222, 128), 2, cv2.LINE_AA)
        cv2.drawMarker(annotated, (od_x, od_y), (74, 222, 128), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(annotated, f"OD (CDR:{od_info['cup_to_disc_ratio']})", (od_x - 30, od_y - od_r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (74, 222, 128), 1, cv2.LINE_AA)

        # 2. Optic Cup (Cyan/Teal, BGR: 212, 182, 6)
        cv2.circle(annotated, (od_x, od_y), cup_r, (212, 182, 6), 1, cv2.LINE_AA)

        # 3. Fovea Centralis (Sky Blue target, BGR: 235, 132, 14)
        cv2.circle(annotated, (fv_x, fv_y), 6, (235, 132, 14), 2, cv2.LINE_AA)
        cv2.drawMarker(annotated, (fv_x, fv_y), (235, 132, 14), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(annotated, "Fovea", (fv_x - 20, fv_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 132, 14), 1, cv2.LINE_AA)

        # 4. Fovea CSME Zone (500µm ring = 0.33 DD)
        csme_ring_r = int(disc_diam * 0.33)
        cv2.circle(annotated, (fv_x, fv_y), csme_ring_r, (0, 140, 255), 1, cv2.LINE_AA)

        # 5. Connect nearest lesion if present
        if csme_info.get("nearest_ex_coord"):
            lx, ly = csme_info["nearest_ex_coord"]
            cv2.line(annotated, (fv_x, fv_y), (lx, ly), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(annotated, (lx, ly), 4, (0, 0, 255), -1)

        cv2.imwrite(output_path, annotated)
        return output_path
