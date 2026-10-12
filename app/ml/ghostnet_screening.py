"""
GhostNet Multi-Disease Screening Architecture and Grad-CAM Saliency Engine.
Implements the GhostNet architecture (Han et al., CVPR 2020) with lightweight
Ghost modules for fast, edge-ready multi-label retinal disease screening.
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCREENING_DISEASES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Hypertensive Retinopathy",
    "Age-Related Macular Degeneration (AMD)",
    "Normal"
]


class SqueezeAndExcite(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, in_chs, se_ratio=0.25):
        super(SqueezeAndExcite, self).__init__()
        reduced_chs = max(1, int(in_chs * se_ratio))
        self.fc1 = nn.Conv2d(in_chs, reduced_chs, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_chs, in_chs, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_x = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return input_x * self.sigmoid(x)


class GhostModule(nn.Module):
    """
    Ghost Module generates more feature maps from cheap linear operations.
    Primary conv generates m features, cheap depthwise conv generates remainder.
    """
    def __init__(self, in_chs, out_chs, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :, :]


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck with residual skip connection and optional SE module."""
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.0):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depthwise convolution if downsampling
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = SqueezeAndExcite(mid_chs, se_ratio) if se_ratio > 0.0 else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # Shortcut path
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs)
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck stage
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck stage
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    """
    Full GhostNet backbone for retinal image representation.
    """
    def __init__(self, width=1.0):
        super(GhostNet, self).__init__()
        stem_out = int(16 * width)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, stem_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True)
        )

        # Ghost bottleneck configuration: [k, exp, c, se, s]
        cfg = [
            # stage 1
            [3, 16, 16, 0, 1],
            # stage 2
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            # stage 3
            [5, 72, 40, 0.25, 2],
            [5, 120, 40, 0.25, 1],
            # stage 4
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
            # stage 5
            [5, 672, 160, 0.25, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1]
        ]

        stages = []
        in_chs = stem_out
        for k, exp, c, se, s in cfg:
            out_chs = int(c * width)
            mid_chs = int(exp * width)
            stages.append(GhostBottleneck(in_chs, mid_chs, out_chs, dw_kernel_size=k, stride=s, se_ratio=se))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stages)

        out_chs = int(960 * width)
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_chs

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        return x


class MultiDiseaseGhostNet(nn.Module):
    """
    GhostNet Multi-Label Retinal Disease Screening Model.
    Produces independent sigmoid probability scores for co-occurring conditions:
    Diabetic Retinopathy, Glaucoma, Cataract, Hypertensive Retinopathy, AMD, Normal.
    """
    def __init__(self, num_classes=len(SCREENING_DISEASES), width=1.0):
        super(MultiDiseaseGhostNet, self).__init__()
        self.backbone = GhostNet(width=width)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        # Hook target layer for Grad-CAM
        self.target_layer = self.backbone.conv_head

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(pooled)
        return logits


class GhostNetGradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM & Grad-CAM++) engine for GhostNet.
    Grad-CAM++ uses second and third-order gradient weighting to achieve superior localization
    of multiple co-occurring retinal pathologies and fine microvascular lesions.
    """
    def __init__(self, model: MultiDiseaseGhostNet, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None, method: str = "gradcam++") -> np.ndarray:
        """
        Generates normalized saliency heatmap (0.0 to 1.0) for the target class using Grad-CAM++.
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            # Pick highest abnormal class or highest overall
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if no gradients
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5

        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        if method == "gradcam++":
            # Grad-CAM++ formulation:
            # alpha = (grads^2) / (2 * grads^2 + sum(acts * grads^3) + eps)
            # weights = sum(alpha * relu(grads))
            grads_pow2 = torch.pow(grads, 2)
            grads_pow3 = torch.pow(grads, 3)
            sum_acts = torch.sum(acts, dim=[2, 3], keepdim=True)
            eps = 1e-7

            alpha_denom = 2.0 * grads_pow2 + sum_acts * grads_pow3
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
            alpha = grads_pow2 / alpha_denom

            weights = torch.sum(alpha * F.relu(grads), dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
        else:
            # Standard Grad-CAM
            weights = torch.mean(grads, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def generate_heatmap_overlay(self, original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: int = cv2.COLORMAP_TURBO) -> tuple[np.ndarray, np.ndarray]:
        """
        Overlays Grad-CAM heatmap on RGB fundus image with specified OpenCV colormap.
        """
        h, w = original_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.float32(heatmap_rgb) * alpha + np.float32(original_rgb) * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay, cam_resized

    def generate_multichannel_overlays(self, original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> dict:
        """
        Produces multiple clinically validated colormaps:
        - Turbo: Modern perceptually uniform high-contrast color scale
        - Jet: Classic clinical spectrum
        - Viridis: Colorblind-safe uniform perceptually linear scale
        """
        h, w = original_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_u8 = np.uint8(255 * cam_resized)

        cmaps = {
            "turbo": cv2.COLORMAP_TURBO,
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO
        }

        overlays = {}
        for name, cmap in cmaps.items():
            heatmap = cv2.applyColorMap(cam_u8, cmap)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            ov = np.float32(heatmap_rgb) * alpha + np.float32(original_rgb) * (1 - alpha)
            overlays[name] = np.clip(ov, 0, 255).astype(np.uint8)

        return overlays, cam_resized
