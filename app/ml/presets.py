import os
import math
import numpy as np
import cv2

PRESET_DIR = "app/static/images/presets"

def draw_retinal_base(width=512, height=512):
    """Create high-fidelity base fundus disc with vascular arcades, optic disc, and macula."""
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

    # Fundus color channels (BGR): Red rich, moderate green, low blue
    b = (20 * (1 - norm_dist * 0.5)).astype(np.uint8)
    g = (60 * (1 - norm_dist * 0.4)).astype(np.uint8)
    r = (185 * (1 - norm_dist * 0.35)).astype(np.uint8)
    base_color = cv2.merge([b, g, r])

    # Apply circular mask
    img = cv2.bitwise_and(base_color, base_color, mask=mask)

    # Optic Disc (typically nasal side: left or right, let's place it at x=140, y=256)
    od_center = (145, 256)
    od_radius = 42
    # Optic disc yellowish-pink rim
    cv2.circle(img, od_center, od_radius, (120, 200, 245), -1)
    # Physiological cup (inner pale yellow)
    cup_radius = int(od_radius * 0.35)
    cv2.circle(img, od_center, cup_radius, (160, 230, 255), -1)

    # Macula / Fovea (temporal side: x=330, y=260)
    macula_center = (330, 260)
    # Macular lutea: slightly darker reddish-brown depression
    overlay = img.copy()
    cv2.circle(overlay, macula_center, 38, (10, 35, 120), -1)
    cv2.circle(overlay, macula_center, 12, (5, 20, 90), -1) # Foveola
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

    # Re-apply circular aperture mask so nothing spills outside
    img = cv2.bitwise_and(img, img, mask=mask)
    return img, od_center, macula_center, mask

def generate_preset_images():
    """Generates all preset fundus images if not already created."""
    os.makedirs(PRESET_DIR, exist_ok=True)

    presets = {
        "sample_normal.jpg": "normal",
        "sample_dr_mild.jpg": "dr_mild",
        "sample_dr_moderate.jpg": "dr_moderate",
        "sample_dr_severe.jpg": "dr_severe",
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
            # Subtle smooth natural texture
            pass

        elif disease_type == "dr_mild":
            # Microaneurysms (tiny red dots in temporal periphery)
            np.random.seed(42)
            for _ in range(18):
                rx = int(np.random.normal(310, 45))
                ry = int(np.random.normal(250, 45))
                if mask[min(ry, 511), min(rx, 511)] > 0:
                    cv2.circle(img, (rx, ry), 2, (10, 15, 95), -1)

        elif disease_type == "dr_moderate":
            # Microaneurysms + dot-blot hemorrhages + hard exudates (bright yellow wax-like deposits)
            np.random.seed(101)
            # Hemorrhages
            for _ in range(35):
                rx = int(np.random.normal(290, 55))
                ry = int(np.random.normal(260, 55))
                if mask[min(ry, 511), min(rx, 511)] > 0:
                    cv2.circle(img, (rx, ry), int(np.random.choice([2, 3, 4])), (10, 15, 85), -1)
            # Hard exudates (bright yellow deposits)
            for _ in range(25):
                ex = int(np.random.normal(270, 35))
                ey = int(np.random.normal(230, 35))
                if mask[min(ey, 511), min(ex, 511)] > 0:
                    cv2.circle(img, (ex, ey), int(np.random.choice([2, 3])), (120, 240, 255), -1)

        elif disease_type == "dr_severe":
            # Confluent blot hemorrhages, cotton-wool spots, dense hard exudate circinate rings
            np.random.seed(777)
            # Hemorrhages
            for _ in range(65):
                rx = int(np.random.normal(280, 75))
                ry = int(np.random.normal(260, 75))
                if mask[min(ry, 511), min(rx, 511)] > 0:
                    cv2.circle(img, (rx, ry), int(np.random.choice([3, 4, 6])), (5, 10, 75), -1)
            # Dense hard exudate clusters
            for _ in range(50):
                ex = int(np.random.normal(320, 40))
                ey = int(np.random.normal(250, 40))
                if mask[min(ey, 511), min(ex, 511)] > 0:
                    cv2.circle(img, (ex, ey), int(np.random.choice([3, 4, 5])), (100, 245, 255), -1)
            # Cotton-wool spots (fluffy pale patches)
            for _ in range(4):
                cx = int(np.random.normal(240, 30))
                cy = int(np.random.normal(200, 30))
                cv2.circle(img, (cx, cy), 10, (190, 215, 230), -1)

        elif disease_type == "glaucoma":
            # Significant cup enlargement (CDR >= 0.75), pallor, bayoneting of vessels
            cv2.circle(img, od, 42, (110, 190, 240), -1)
            # Enlarged cup (CDR 0.78)
            cv2.circle(img, od, 33, (180, 240, 255), -1)
            # Peripapillary atrophy crescent
            cv2.ellipse(img, od, (48, 44), 30, 0, 140, (100, 160, 210), 4)

        elif disease_type == "cataract":
            # Diffuse lens haze/blur, reduced contrast, brownish-yellow nuclear cataract veil
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
            haze = np.full_like(img, (20, 70, 140), dtype=np.uint8) # Amber haze
            img = cv2.addWeighted(blurred, 0.65, haze, 0.35, 0)
            img = cv2.bitwise_and(img, img, mask=mask)

        elif disease_type == "amd":
            # Age-Related Macular Degeneration: Drusen deposits (yellowish-white spots in macula)
            np.random.seed(555)
            # Soft & hard drusen clustered in macula
            for _ in range(45):
                dx = int(np.random.normal(macula[0], 28))
                dy = int(np.random.normal(macula[1], 28))
                if mask[min(dy, 511), min(dx, 511)] > 0:
                    cv2.circle(img, (dx, dy), int(np.random.choice([3, 4, 5])), (110, 220, 250), -1)
            # Macular pigment mottling
            for _ in range(20):
                px = int(np.random.normal(macula[0], 20))
                py = int(np.random.normal(macula[1], 20))
                if mask[min(py, 511), min(px, 511)] > 0:
                    cv2.circle(img, (px, py), 2, (15, 20, 80), -1)

        elif disease_type == "hypertensive":
            # Arteriolar narrowing, copper wiring, AV nicking
            # Subtle flame-shaped hemorrhages
            np.random.seed(333)
            for _ in range(12):
                hx = int(np.random.normal(260, 40))
                hy = int(np.random.normal(210, 40))
                if mask[min(hy, 511), min(hx, 511)] > 0:
                    cv2.line(img, (hx, hy), (hx + 8, hy + 4), (10, 15, 90), 3)

        # Smooth realistic blending
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(filepath, img)

    print("Presets verified/generated in:", PRESET_DIR)

if __name__ == "__main__":
    generate_preset_images()
