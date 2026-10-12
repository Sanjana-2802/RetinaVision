"""
Clinical Retinal Fundus Presets Generator.
Generates realistic benchmark fundus scans covering:
1. Normal Healthy Retina (Grade 0)
2. Mild NPDR (Grade 1 - Microaneurysms)
3. Moderate NPDR (Grade 2 - Dot/Blot Hemorrhages + Hard Exudates)
4. Severe NPDR (Grade 3 - Extensive Hemorrhages + Cotton Wool Spots)
5. Proliferative DR (Grade 4 - Neovascularization + Preretinal Hemorrhage)
6. Glaucoma (Enlarged Cup CDR 0.78 + Peripapillary Atrophy)
7. Cataract (Nuclear Lens Opacification)
8. Hypertensive Retinopathy (Flame Hemorrhages + Copper Wiring)
9. Age-Related Macular Degeneration (Macular Drusen)
"""

import os
import math
import numpy as np
import cv2

PRESET_DIR = "app/static/images/presets"


def draw_retinal_base(width=512, height=512):
    """Creates a high-fidelity base fundus disc with vascular arcades, optic disc, and macula."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(width, height) * 0.46)

    # Circular mask for retinal fundus
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Realistic reddish-orange choroidal gradient
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    norm_dist = np.clip(dist_from_center / radius, 0, 1)

    b = (22 * (1 - norm_dist * 0.5)).astype(np.uint8)
    g = (65 * (1 - norm_dist * 0.4)).astype(np.uint8)
    r = (190 * (1 - norm_dist * 0.35)).astype(np.uint8)
    base_color = cv2.merge([b, g, r])

    # Apply circular mask
    img = cv2.bitwise_and(base_color, base_color, mask=mask)

    # Optic Disc (nasal side: x=145, y=256)
    od_center = (145, 256)
    od_radius = 42
    # Optic disc yellowish-pink rim
    cv2.circle(img, od_center, od_radius, (120, 205, 245), -1)
    # Physiological cup (inner pale yellow, normal CDR ~ 0.35)
    cup_radius = int(od_radius * 0.35)
    cv2.circle(img, od_center, cup_radius, (160, 230, 255), -1)

    # Macula / Fovea (temporal side: x=330, y=260)
    macula_center = (330, 260)
    overlay = img.copy()
    cv2.circle(overlay, macula_center, 38, (10, 35, 120), -1)
    cv2.circle(overlay, macula_center, 12, (5, 20, 90), -1)  # Foveola
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Retinal Blood Vessel Tree branching from Optic Disc
    def draw_vessel_branch(start_pt, angles, lengths, thicknesses, color=(15, 25, 120)):
        curr_pt = start_pt
        for ang, length, thick in zip(angles, lengths, thicknesses):
            rad = math.radians(ang)
            next_pt = (int(curr_pt[0] + length * math.cos(rad)), int(curr_pt[1] + length * math.sin(rad)))
            cv2.line(img, curr_pt, next_pt, color, thick, cv2.LINE_AA)
            curr_pt = next_pt

    # Superior temporal arcade
    draw_vessel_branch((145, 240), [-60, -40, -15, 5, 20], [35, 45, 50, 60, 50], [4, 3, 3, 2, 1])
    draw_vessel_branch((175, 205), [-30, -5, 10], [30, 40, 50], [2, 2, 1])
    # Inferior temporal arcade
    draw_vessel_branch((145, 270), [60, 40, 15, -5, -20], [35, 45, 50, 60, 50], [4, 3, 3, 2, 1])
    draw_vessel_branch((175, 305), [30, 5, -10], [30, 40, 50], [2, 2, 1])
    # Nasal branches
    draw_vessel_branch((130, 245), [-135, -150, -170], [30, 35, 40], [3, 2, 1])
    draw_vessel_branch((130, 265), [135, 150, 170], [30, 35, 40], [3, 2, 1])

    img = cv2.bitwise_and(img, img, mask=mask)
    return img, od_center, macula_center, mask


def generate_preset_images():
    """Generates all preset fundus benchmark images if not already present."""
    os.makedirs(PRESET_DIR, exist_ok=True)

    presets = {
        "sample_normal.jpg": "normal",
        "sample_dr_mild.jpg": "dr_mild",
        "sample_dr_moderate.jpg": "dr_moderate",
        "sample_dr_severe.jpg": "dr_severe",
        "sample_dr_proliferative.jpg": "dr_proliferative",
        "sample_glaucoma.jpg": "glaucoma",
        "sample_cataract.jpg": "cataract",
        "sample_amd.jpg": "amd",
        "sample_hypertensive.jpg": "hypertensive"
    }

    for filename, disease_type in presets.items():
        filepath = os.path.join(PRESET_DIR, filename)
        if os.path.exists(filepath):
            continue

        base_img, od, macula, mask = draw_retinal_base()
        img = base_img.copy()

        if disease_type == "normal":
            pass

        elif disease_type == "dr_mild":
            # Microaneurysms only (isolated tiny red dots)
            np.random.seed(42)
            for _ in range(18):
                rx = int(np.random.normal(310, 45))
                ry = int(np.random.normal(250, 45))
                if 0 <= ry < 512 and 0 <= rx < 512 and mask[ry, rx] > 0:
                    cv2.circle(img, (rx, ry), 2, (10, 15, 95), -1)

        elif disease_type == "dr_moderate":
            # Microaneurysms + dot-blot hemorrhages + hard exudates
            np.random.seed(101)
            # Dot-blot hemorrhages
            for _ in range(35):
                rx = int(np.random.normal(290, 55))
                ry = int(np.random.normal(260, 55))
                if 0 <= ry < 512 and 0 <= rx < 512 and mask[ry, rx] > 0:
                    cv2.circle(img, (rx, ry), int(np.random.choice([2, 3, 4])), (10, 15, 85), -1)
            # Hard exudates (waxy yellow deposits)
            for _ in range(25):
                ex = int(np.random.normal(270, 35))
                ey = int(np.random.normal(230, 35))
                if 0 <= ey < 512 and 0 <= ex < 512 and mask[ey, ex] > 0:
                    cv2.circle(img, (ex, ey), int(np.random.choice([2, 3])), (120, 240, 255), -1)

        elif disease_type == "dr_severe":
            # Extensive 4-quadrant intraretinal hemorrhages + cotton wool spots + dense exudates
            np.random.seed(777)
            # Blot hemorrhages in all quadrants
            for _ in range(68):
                rx = int(np.random.normal(280, 85))
                ry = int(np.random.normal(260, 85))
                if 0 <= ry < 512 and 0 <= rx < 512 and mask[ry, rx] > 0:
                    cv2.circle(img, (rx, ry), int(np.random.choice([3, 4, 6])), (5, 10, 75), -1)
            # Hard exudates
            for _ in range(45):
                ex = int(np.random.normal(320, 45))
                ey = int(np.random.normal(250, 45))
                if 0 <= ey < 512 and 0 <= ex < 512 and mask[ey, ex] > 0:
                    cv2.circle(img, (ex, ey), int(np.random.choice([3, 4, 5])), (100, 245, 255), -1)
            # Soft exudates (cotton-wool spots: fluffy pale patches)
            for _ in range(5):
                cx = int(np.random.normal(240, 30))
                cy = int(np.random.normal(200, 30))
                if 0 <= cy < 512 and 0 <= cx < 512 and mask[cy, cx] > 0:
                    cv2.circle(img, (cx, cy), 11, (190, 215, 230), -1)

        elif disease_type == "dr_proliferative":
            # Grade 4 PDR: Neovascularization fronds, large preretinal/vitreous bleeds, dense lesions
            np.random.seed(999)
            # Large preretinal boat-shaped hemorrhage
            pts = np.array([[220, 310], [310, 320], [300, 360], [230, 355]], np.int32)
            cv2.fillPoly(img, [pts], (5, 5, 65))
            # Neovascular fronds: fine abnormal branching vessels near disc
            for angle in np.linspace(-60, 60, 8):
                rad = math.radians(angle)
                end_x = int(od[0] + 55 * math.cos(rad))
                end_y = int(od[1] + 55 * math.sin(rad))
                cv2.line(img, od, (end_x, end_y), (10, 20, 110), 1, cv2.LINE_AA)
            # Scattered hemorrhages
            for _ in range(55):
                rx = int(np.random.normal(270, 75))
                ry = int(np.random.normal(260, 75))
                if 0 <= ry < 512 and 0 <= rx < 512 and mask[ry, rx] > 0:
                    cv2.circle(img, (rx, ry), int(np.random.choice([3, 5])), (5, 10, 70), -1)
            # Cotton wool spots
            for _ in range(4):
                cx = int(np.random.normal(230, 25))
                cy = int(np.random.normal(190, 25))
                if 0 <= cy < 512 and 0 <= cx < 512 and mask[cy, cx] > 0:
                    cv2.circle(img, (cx, cy), 12, (195, 220, 235), -1)

        elif disease_type == "glaucoma":
            # Significant optic cup enlargement (CDR >= 0.78), pallor, rim thinning
            cv2.circle(img, od, 42, (110, 190, 240), -1)
            cv2.circle(img, od, 33, (180, 240, 255), -1)  # Enlarged cup
            cv2.ellipse(img, od, (48, 44), 30, 0, 140, (100, 160, 210), 4)

        elif disease_type == "cataract":
            # Nuclear lens haze, reduced contrast, yellowish blur
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            haze = np.full_like(img, (20, 70, 140), dtype=np.uint8)
            img = cv2.addWeighted(blurred, 0.65, haze, 0.35, 0)
            img = cv2.bitwise_and(img, img, mask=mask)

        elif disease_type == "amd":
            # Confluent drusen deposits clustered in macula
            np.random.seed(555)
            for _ in range(45):
                dx = int(np.random.normal(macula[0], 28))
                dy = int(np.random.normal(macula[1], 28))
                if 0 <= dy < 512 and 0 <= dx < 512 and mask[dy, dx] > 0:
                    cv2.circle(img, (dx, dy), int(np.random.choice([3, 4, 5])), (110, 220, 250), -1)

        elif disease_type == "hypertensive":
            # Arteriolar narrowing, flame hemorrhages
            np.random.seed(333)
            for _ in range(14):
                hx = int(np.random.normal(260, 40))
                hy = int(np.random.normal(210, 40))
                if 0 <= hy < 512 and 0 <= hx < 512 and mask[hy, hx] > 0:
                    cv2.line(img, (hx, hy), (hx + 10, hy + 5), (10, 15, 90), 3)

        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(filepath, img)

    print("Preset images generated successfully in:", PRESET_DIR)


if __name__ == "__main__":
    generate_preset_images()
