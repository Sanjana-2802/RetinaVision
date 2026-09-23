import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Explainable AI (XAI) Grad-CAM engine for Retinal Deep Learning Models.
    Computes class activation maps that highlight pathognomonic lesions
    (exudates, hemorrhages, optic cup cupping, drusen).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Calculates normalized Grad-CAM activation map (0.0 to 1.0)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        logits, _ = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback uniform activation map if gradients not caught
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Only features with positive influence

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def create_overlay(self, original_rgb: np.ndarray, heatmap_2d: np.ndarray, alpha: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms raw 2D CAM into a clinical colormap (Jet / Turbo) and blends with the fundus image.
        Returns:
            - color_heatmap (RGB)
            - blended_overlay (RGB)
        """
        h, w = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap_2d, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Color map (COLORMAP_JET)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend onto original fundus image
        blended = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return heatmap_rgb, blended

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
