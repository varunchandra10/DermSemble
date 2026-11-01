import json
from pathlib import Path
import onnxruntime as ort
from backend.core.config import settings
from backend.core.logger import log
from backend.utils.exceptions import ModelNotLoadedError


class ModelLoader:
    """
    Singleton loader managing production INT8 ONNX sessions and clinical manifest.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.convnext_session: ort.InferenceSession | None = None
        self.swin_session: ort.InferenceSession | None = None
        self.manifest: dict | None = None
        
        self.label_map: dict[int, str] = {}
        self.id_map: dict[str, int] = {}
        self.clinical_tiers: dict[str, str] = {}
        self.w_conv: float = settings.W_CONVNEXT
        self.w_swin: float = settings.W_SWIN
        self.alpha_mel: float = settings.MELANOMA_SAFETY_MULTIPLIER
        self.theta_abstain: float = settings.ABSTENTION_THRESHOLD
        
        self._initialized = True

    def load_all(self):
        """Loads manifest and initializes ONNX Runtime sessions."""
        log.info("Loading Dermsemble Production Manifest & ONNX INT8 Models...")
        self._load_manifest()
        self._load_onnx_sessions()
        log.info("[OK] Dermsemble Production Pipeline initialized successfully.")

    def _load_manifest(self):
        manifest_path = settings.DEPLOYMENT_MANIFEST_PATH
        try:
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                
                raw_labels = self.manifest.get("label_map", {})
                self.label_map = {int(k): str(v) for k, v in raw_labels.items()}
                self.id_map = {v: k for k, v in self.label_map.items()}
                self.clinical_tiers = self.manifest.get("clinical_tiers", {})
                
                weights = self.manifest.get("ensemble_weights", {})
                self.w_conv = float(weights.get("convnext_tiny", settings.W_CONVNEXT))
                self.w_swin = float(weights.get("swin_transformer_tiny", settings.W_SWIN))
                
                policies = self.manifest.get("decision_policies", {})
                self.alpha_mel = float(policies.get("melanoma_safety_multiplier_alpha", settings.MELANOMA_SAFETY_MULTIPLIER))
                self.theta_abstain = float(policies.get("clinical_abstention_threshold_theta", settings.ABSTENTION_THRESHOLD))
                
                log.info(
                    f"[OK] Manifest loaded: {len(self.label_map)} classes, "
                    f"Weights=(Conv: {self.w_conv}, Swin: {self.w_swin}), "
                    f"Alpha={self.alpha_mel}, Theta={self.theta_abstain}"
                )
            else:
                log.error(f"Manifest missing at {manifest_path}")
        except Exception as e:
            log.error(f"Error parsing deployment manifest: {e}")

    def _load_onnx_sessions(self):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # 1. ConvNeXt-Tiny INT8 Session
        try:
            conv_path = settings.CONVNEXT_INT8_ONNX_PATH
            if conv_path.exists():
                self.convnext_session = ort.InferenceSession(
                    str(conv_path),
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                log.info(f"[OK] ConvNeXt-Tiny INT8 Session active: {conv_path.name}")
            else:
                log.error(f"ConvNeXt-Tiny ONNX missing at {conv_path}")
        except Exception as e:
            log.error(f"Failed to load ConvNeXt-Tiny ONNX: {e}")

        # 2. Swin-Transformer Tiny INT8 Session
        try:
            swin_path = settings.SWIN_INT8_ONNX_PATH
            if swin_path.exists():
                self.swin_session = ort.InferenceSession(
                    str(swin_path),
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
                log.info(f"[OK] Swin-Transformer INT8 Session active: {swin_path.name}")
            else:
                log.error(f"Swin-Transformer ONNX missing at {swin_path}")
        except Exception as e:
            log.error(f"Failed to load Swin-Transformer ONNX: {e}")

    def get_convnext_session(self) -> ort.InferenceSession:
        if self.convnext_session is None:
            raise ModelNotLoadedError("ConvNeXt-Tiny INT8 ONNX Session")
        return self.convnext_session

    # Backwards compatibility alias
    def get_cnn_session(self) -> ort.InferenceSession:
        return self.get_convnext_session()

    def get_swin_session(self) -> ort.InferenceSession:
        if self.swin_session is None:
            raise ModelNotLoadedError("Swin-Transformer INT8 ONNX Session")
        return self.swin_session

    def get_label(self, idx: int) -> str:
        return self.label_map.get(idx, f"Unknown_Class_{idx}")

    def get_class_id(self, label: str) -> int:
        return self.id_map.get(label, -1)

    def get_tier(self, label: str) -> str:
        return self.clinical_tiers.get(label, "Tier 2")


model_loader = ModelLoader()
