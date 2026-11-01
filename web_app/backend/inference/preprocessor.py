import cv2
import numpy as np
from PIL import Image
from backend.core.config import settings
from backend.utils.image_utils import ensure_rgb


class ImagePreprocessor:
    """
    Production-grade Clinical Image Preprocessor.
    Implements:
      1. DullRazor: Multi-directional hair artifact removal + Telea inpainting.
      2. Shades-of-Gray: Minkowski p-norm (p=6) illumination color constancy.
      3. Standardized 384x384 spatial scaling.
      4. ImageNet channel normalization for ONNX tensor ingestion.
    """
    def __init__(self):
        self.target_size = (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
        self.mean = np.array(settings.NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(settings.NORM_STD, dtype=np.float32).reshape(1, 1, 3)

    @staticmethod
    def refined_dull_razor(img_bgr: np.ndarray, threshold: int = 22, kernel_len: int = 17, inpaint_radius: int = 2) -> np.ndarray:
        """Removes fine dark hairs using directional morphological kernels without blurring lesion edges."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        diff_accum = np.zeros_like(gray)
        angles = [0, 45, 90, 135]
        
        for angle in angles:
            k = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
            if angle == 0:
                k[kernel_len // 2, :] = 1
            elif angle == 90:
                k[:, kernel_len // 2] = 1
            elif angle == 45:
                np.fill_diagonal(k, 1)
            elif angle == 135:
                np.fill_diagonal(np.fliplr(k), 1)
                
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
            diff = cv2.subtract(closed, gray)
            diff_accum = np.maximum(diff_accum, diff)
            
        _, raw_mask = cv2.threshold(diff_accum, threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
        clean_mask = np.zeros_like(raw_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 8 <= area <= 1200:
                clean_mask[labels == i] = 255
                
        dilated_mask = cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if np.sum(dilated_mask) > 50:
            return cv2.inpaint(img_bgr, dilated_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
        return img_bgr

    @staticmethod
    def shades_of_gray(img_rgb: np.ndarray, p: int = 6) -> np.ndarray:
        """Applies Minkowski p-norm (p=6) gray-world color constancy to neutralize lighting casts."""
        img_f = img_rgb.astype(np.float32)
        h, w = img_rgb.shape[:2]
        step = max(1, int(max(h, w) / 256))
        sub = img_f[::step, ::step]
        
        norm_r = float((np.mean(sub[:, :, 0] ** p)) ** (1.0 / p))
        norm_g = float((np.mean(sub[:, :, 1] ** p)) ** (1.0 / p))
        norm_b = float((np.mean(sub[:, :, 2] ** p)) ** (1.0 / p))
        
        norm_mean = (norm_r + norm_g + norm_b) / 3.0
        kr = norm_mean / (norm_r + 1e-5)
        kg = norm_mean / (norm_g + 1e-5)
        kb = norm_mean / (norm_b + 1e-5)
        
        img_corr = np.zeros_like(img_f)
        img_corr[:, :, 0] = np.clip(img_f[:, :, 0] * kr, 0.0, 255.0)
        img_corr[:, :, 1] = np.clip(img_f[:, :, 1] * kg, 0.0, 255.0)
        img_corr[:, :, 2] = np.clip(img_f[:, :, 2] * kb, 0.0, 255.0)
        return img_corr.astype(np.uint8)

    def preprocess_image(self, pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """
        Full clinical preprocessing pipeline.
        
        Returns:
            tensor_input: np.ndarray of shape (1, 3, 384, 384) in float32
            clean_rgb_384: np.ndarray of shape (384, 384, 3) in uint8 for heatmap overlays
        """
        # Ensure RGB
        img_rgb = ensure_rgb(pil_img)
        arr_rgb = np.array(img_rgb)
        
        # 1. DullRazor Artifact Removal
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        clean_bgr = self.refined_dull_razor(arr_bgr)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Shades-of-Gray Color Constancy
        calibrated_rgb = self.shades_of_gray(clean_rgb, p=6)
        
        # 3. Standardized 384x384 Resizing
        target_w, target_h = self.target_size
        h, w = calibrated_rgb.shape[:2]
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LANCZOS4
        clean_rgb_384 = cv2.resize(calibrated_rgb, (target_w, target_h), interpolation=interp)
        
        # 4. Normalize for ONNX Ingestion (1, 3, 384, 384)
        norm_img = clean_rgb_384.astype(np.float32) / 255.0
        norm_img = (norm_img - self.mean) / self.std
        tensor = np.transpose(norm_img, (2, 0, 1))  # (3, H, W)
        tensor = np.expand_dims(tensor, axis=0)      # (1, 3, H, W)
        
        return tensor.astype(np.float32), clean_rgb_384

    # Backwards compatibility wrappers
    def preprocess_for_dl(self, pil_img: Image.Image) -> np.ndarray:
        tensor, _ = self.preprocess_image(pil_img)
        return tensor

    def preprocess_for_xgb(self, pil_img: Image.Image) -> Image.Image:
        _, clean_rgb = self.preprocess_image(pil_img)
        return Image.fromarray(clean_rgb)


preprocessor = ImagePreprocessor()
