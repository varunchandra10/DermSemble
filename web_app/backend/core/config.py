import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Resolution
BACKEND_DIR = Path(__file__).resolve().parent.parent
WEB_APP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEB_APP_DIR.parent
ONNX_DIR = PROJECT_ROOT / "Models" / "onnx"
CHECKPOINTS_DIR = PROJECT_ROOT / "Models" / "checkpoints"
LOCAL_MODELS_DIR = BACKEND_DIR / "models"

def _resolve_binary_path(filename: str) -> Path:
    primary = ONNX_DIR / filename
    if primary.exists():
        return primary
    fallback = LOCAL_MODELS_DIR / filename
    if fallback.exists():
        return fallback
    return primary


class Settings(BaseSettings):
    """
    Centralized configuration for Dermsemble FastAPI Production Backend.
    """
    # Application identity
    APP_NAME: str = "Dermsemble Diagnostic API"
    APP_VERSION: str = "2.0.0-production"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Inference & Preprocessing Specifications
    IMAGE_SIZE: int = 384  # Standardized 384x384 Native Resolution
    BATCH_SIZE: int = 16
    MAX_UPLOAD_SIZE: int = 25 * 1024 * 1024  # 25 MB limit for high-res clinical photos

    # Normalization constants (ImageNet standard)
    NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    NORM_STD: list[float] = [0.229, 0.224, 0.225]

    # Directories
    PROJECT_ROOT_PATH: Path = PROJECT_ROOT
    ONNX_DIR_PATH: Path = ONNX_DIR
    STATIC_DIR: Path = BACKEND_DIR / "static"
    HEATMAPS_DIR: Path = BACKEND_DIR / "static" / "heatmaps"

    # Production Model and Manifest Binaries
    DEPLOYMENT_MANIFEST_PATH: Path = _resolve_binary_path("production_deployment_manifest.json")
    CONVNEXT_INT8_ONNX_PATH: Path = _resolve_binary_path("convnext_tiny_int8.onnx")
    SWIN_INT8_ONNX_PATH: Path = _resolve_binary_path("swin_tiny_int8.onnx")
    
    # Optional PyTorch Checkpoints (for Grad-CAM feature attribution)
    CONVNEXT_PT_CKPT: Path = CHECKPOINTS_DIR / "convnext_tiny" / "best_convnext_tiny.pth"
    SWIN_PT_CKPT: Path = CHECKPOINTS_DIR / "swin_tiny" / "best_swin_tiny.pth"

    # Default Clinical Safety Parameters (dynamically loaded from manifest)
    MELANOMA_SAFETY_MULTIPLIER: float = 1.85
    ABSTENTION_THRESHOLD: float = 0.50
    W_CONVNEXT: float = 0.42
    W_SWIN: float = 0.58

    # CORS Settings
    CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
os.makedirs(settings.HEATMAPS_DIR, exist_ok=True)
