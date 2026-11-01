import numpy as np
from backend.inference.model_loader import model_loader
from backend.core.logger import log

class Calibrator:
    """
    Applies calibration to the raw probabilities.
    Includes temperature scaling and clipping.
    """
    def __init__(self):
        # Default temperature if thresholds file is missing or doesn't have it
        self.temperature = 1.0 
        self._load_config()

    def _load_config(self):
        try:
            thresholds = model_loader.thresholds
            if thresholds and "temperature" in thresholds:
                self.temperature = thresholds["temperature"]
        except Exception:
            pass

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Applies temperature scaling.
        probabilities shape: (1, num_classes)
        """
        # Convert probabilities back to approximate logits for temperature scaling
        # (This assumes probabilities haven't been severely clipped yet)
        epsilon = 1e-7
        probs_clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        logits = np.log(probs_clipped)
        
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Softmax again
        e_x = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        calibrated_probs = e_x / e_x.sum(axis=1, keepdims=True)
        
        return calibrated_probs

calibrator = Calibrator()
