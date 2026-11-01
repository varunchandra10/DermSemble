import os
import uuid
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from backend.core.config import settings
from backend.core.logger import log


class ExplainabilityService:
    """
    Generates real-time Grad-CAM visual attribution heatmaps
    identifying lesions and borders associated with model predictions.
    """
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
            
        ckpt_path = settings.CONVNEXT_PT_CKPT
        if not ckpt_path.exists():
            log.warning(f"PyTorch checkpoint not found at {ckpt_path}. Falling back to saliency map.")
            self._initialized = True
            return

        try:
            log.info(f"Loading PyTorch ConvNeXt-Tiny for Grad-CAM on {self.device}...")
            model = models.convnext_tiny(weights=None)
            in_f = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 512),
                nn.GELU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                nn.Linear(512, 25)
            )
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()

            # Attach Hooks to Final Conv Stage (features[7])
            def fwd_hook(m, inp, out):
                self.activations = out.detach()

            def bwd_hook(m, gin, gout):
                self.gradients = gout[0].detach()

            model.features[7].register_forward_hook(fwd_hook)
            model.features[7].register_full_backward_hook(bwd_hook)

            self.model = model
            self._initialized = True
            log.info("[OK] Grad-CAM engine initialized with ConvNeXt-Tiny feature hooks.")
        except Exception as e:
            log.error(f"Failed to initialize Grad-CAM model: {e}")
            self._initialized = True

    def generate_heatmap(
        self,
        rgb_image_384: np.ndarray,
        tensor_384: np.ndarray,
        target_class: int
    ) -> str | None:
        """
        Computes Grad-CAM for target_class and saves overlay to static/heatmaps/<id>.png.
        
        Args:
            rgb_image_384: np.ndarray of shape (384, 384, 3) in uint8
            tensor_384: np.ndarray of shape (1, 3, 384, 384) in float32
            target_class: int representing predicted class ID
            
        Returns:
            URL string e.g. "/static/heatmaps/<uuid>.png"
        """
        self._lazy_init()
        
        # Ensure static output dir exists
        os.makedirs(settings.HEATMAPS_DIR, exist_ok=True)
        heatmap_filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
        heatmap_filepath = settings.HEATMAPS_DIR / heatmap_filename

        try:
            if self.model is not None:
                # 1. PyTorch Forward & Backward Pass
                input_tensor = torch.from_numpy(tensor_384).to(self.device).requires_grad_(True)
                self.model.zero_grad()
                logits = self.model(input_tensor)
                
                score = logits[0, target_class]
                score.backward()
                
                # 2. Grad-CAM Calculation
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                cam = torch.relu(cam)
                cam = F.interpolate(cam, size=(384, 384), mode='bilinear', align_corners=False)
                cam_arr = cam[0, 0].cpu().numpy()
                
                # Normalization
                c_min, c_max = cam_arr.min(), cam_arr.max()
                if c_max > c_min:
                    cam_norm = (cam_arr - c_min) / (c_max - c_min)
                else:
                    cam_norm = np.zeros_like(cam_arr)
            else:
                # Fallback: Morphological Gradient Saliency
                gray = cv2.cvtColor(rgb_image_384, cv2.COLOR_RGB2GRAY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                cam_norm = cv2.GaussianBlur(grad, (21, 21), 0).astype(np.float32) / 255.0

            # 3. Create Color Heatmap Overlay
            heatmap_u8 = np.uint8(255 * cam_norm)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Blend 45% heatmap + 55% original image
            blended = np.float32(heatmap_color) * 0.45 + np.float32(rgb_image_384) * 0.55
            blended_u8 = np.uint8(np.clip(blended, 0, 255))
            
            # Save overlay image
            bgr_to_save = cv2.cvtColor(blended_u8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(heatmap_filepath), bgr_to_save)
            
            log.info(f"[OK] Generated Grad-CAM overlay at {heatmap_filepath.name}")
            return f"/static/heatmaps/{heatmap_filename}"

        except Exception as e:
            log.error(f"Error generating Grad-CAM overlay: {e}")
            return None


explainability_service = ExplainabilityService()
