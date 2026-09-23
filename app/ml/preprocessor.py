import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Standard ImageNet transform for deep learning inference
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def crop_retina_circle(image_bgr: np.ndarray, tolerance: int = 10) -> np.ndarray:
    """Find the circular retinal area and crop excess black/dark margins."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > tolerance
    if not np.any(mask):
        return image_bgr
    
    # Bounding box of retinal region
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    cropped = image_bgr[y0:y1, x0:x1]
    return cropped

def apply_clahe_enhancement(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance retinal microvasculature, exudates, and optic disc boundaries.
    The Green channel is primarily enhanced as it contains the highest contrast for fundus images.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def preprocess_fundus_image(image_path: str, output_path: str = None) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Complete fundus preprocessing:
    1. Reads image
    2. Crops to retinal boundary
    3. Resizes to standard clinical resolution (512x512)
    4. Computes CLAHE enhanced diagnostic visualization
    5. Returns PyTorch input tensor, original RGB array, and enhanced RGB array
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read fundus image at {image_path}")

    # Crop dark margins
    cropped_bgr = crop_retina_circle(img_bgr)
    resized_bgr = cv2.resize(cropped_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    enhanced_bgr = apply_clahe_enhancement(resized_bgr)

    # Save preprocessed image if path specified
    if output_path:
        cv2.imwrite(output_path, enhanced_bgr)

    # Convert to RGB for PIL and PyTorch
    original_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(enhanced_rgb)
    input_tensor = transform_pipeline(pil_img).unsqueeze(0)  # Shape: (1, 3, 256, 256)

    return input_tensor, original_rgb, enhanced_rgb
