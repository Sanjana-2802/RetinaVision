"""
Fundus Image Preprocessing Pipeline with CLAHE Enhancement and Circular Mask Isolation.
Standardizes clinical fundus scans, extracts the active retinal field of view,
and enhances vascular and lesion contrast via green-channel CLAHE.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

convnext_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def crop_retina_circle(image_bgr: np.ndarray, tolerance: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Find the circular retinal area, crop excess black/dark margins, and return circular mask."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    if not np.any(mask):
        h, w = image_bgr.shape[:2]
        full_mask = np.ones((h, w), dtype=np.uint8) * 255
        return image_bgr, full_mask

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = image_bgr[y0:y1, x0:x1]
    cropped_mask = (mask[y0:y1, x0:x1]).astype(np.uint8) * 255
    return cropped, cropped_mask


def apply_clahe_enhancement(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances microvasculature, microaneurysms, hemorrhages, and exudates.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Further boost the green channel (contains the sharpest contrast for lesions)
    b_ch, g_ch, r_ch = cv2.split(enhanced_bgr)
    clahe_green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe_green.apply(g_ch)
    enhanced_bgr = cv2.merge((b_ch, g_enhanced, r_ch))

    return enhanced_bgr


def generate_red_free_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Simulates clinical Red-Free (540nm green-absorption) fundus photography.
    In ophthalmic diagnostics, red-free imaging uses green light where hemoglobin
    strongly absorbs photons, causing microaneurysms, hemorrhages, and vascular
    structures to appear with maximum contrast as dark silhouette features.
    """
    b, g, r = cv2.split(image_bgr)

    # Red-free isolation primarily leverages the green channel with blue microvascular boost
    green_boosted = cv2.addWeighted(g, 0.85, b, 0.15, 0)

    # Localized adaptive contrast enhancement for capillary walls
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(green_boosted)

    # Unsharp mask to highlight subtle microvascular microaneurysms
    blurred = cv2.GaussianBlur(g_clahe, (0, 0), sigmaX=2.0)
    unsharp = cv2.addWeighted(g_clahe, 1.4, blurred, -0.4, 0)

    # Replicate into 3-channel grayscale / subtle clinical teal-green fundus tint
    # Red-free is viewed as high-contrast monochrome with subtle cyan tint
    rf_b = np.clip(unsharp.astype(np.float32) * 1.02, 0, 255).astype(np.uint8)
    rf_g = unsharp
    rf_r = np.clip(unsharp.astype(np.float32) * 0.90, 0, 255).astype(np.uint8)

    red_free_bgr = cv2.merge((rf_b, rf_g, rf_r))
    return red_free_bgr


def preprocess_fundus_image(image_path: str, output_path: str = None) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Legacy compatible preprocessing:
    Returns (input_tensor_256, original_rgb_512, enhanced_rgb_512).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read fundus image at {image_path}")

    cropped_bgr, _ = crop_retina_circle(img_bgr)
    resized_bgr = cv2.resize(cropped_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    enhanced_bgr = apply_clahe_enhancement(resized_bgr)

    if output_path:
        cv2.imwrite(output_path, enhanced_bgr)

    original_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(enhanced_rgb)
    input_tensor = transform_pipeline(pil_img).unsqueeze(0)

    return input_tensor, original_rgb, enhanced_rgb


def preprocess_fundus_complete(image_path: str, output_path: str = None, redfree_path: str = None) -> dict:
    """
    Comprehensive multi-model preprocessing:
    Returns dict containing tensors for GhostNet (256x256), ConvNeXt (224x224), U-Net (256x256),
    plus full-resolution 512x512 original, enhanced, and red-free RGB/BGR arrays, and retinal mask.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read fundus image at {image_path}")

    cropped_bgr, cropped_mask = crop_retina_circle(img_bgr)
    resized_bgr = cv2.resize(cropped_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    retinal_mask = cv2.resize(cropped_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Ensure circular boundary
    h, w = resized_bgr.shape[:2]
    circ_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circ_mask, (w // 2, h // 2), int(min(h, w) * 0.47), 255, -1)
    retinal_mask = cv2.bitwise_and(retinal_mask, circ_mask)

    enhanced_bgr = apply_clahe_enhancement(resized_bgr)
    red_free_bgr = generate_red_free_image(resized_bgr)

    # Mask out background
    resized_bgr = cv2.bitwise_and(resized_bgr, resized_bgr, mask=retinal_mask)
    enhanced_bgr = cv2.bitwise_and(enhanced_bgr, enhanced_bgr, mask=retinal_mask)
    red_free_bgr = cv2.bitwise_and(red_free_bgr, red_free_bgr, mask=retinal_mask)

    if output_path:
        cv2.imwrite(output_path, enhanced_bgr)

    if redfree_path:
        cv2.imwrite(redfree_path, red_free_bgr)

    original_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    red_free_rgb = cv2.cvtColor(red_free_bgr, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(enhanced_rgb)
    tensor_256 = transform_pipeline(pil_img).unsqueeze(0)
    tensor_224 = convnext_pipeline(pil_img).unsqueeze(0)

    return {
        "tensor_256": tensor_256,
        "tensor_224": tensor_224,
        "original_rgb": original_rgb,
        "enhanced_rgb": enhanced_rgb,
        "red_free_rgb": red_free_rgb,
        "original_bgr": resized_bgr,
        "enhanced_bgr": enhanced_bgr,
        "red_free_bgr": red_free_bgr,
        "retinal_mask": retinal_mask
    }
