import numpy as np
from backend.inference.model_loader import model_loader
from backend.inference.cnn_predictor import softmax
from backend.utils.exceptions import InferenceError
from backend.core.logger import log

class SwinPredictor:
    """
    Handles inference using the Swin Transformer ONNX model.
    """
    def predict(self, tensor_img: np.ndarray) -> np.ndarray:
        try:
            session = model_loader.get_swin_session()
            
            input_name = session.get_inputs()[0].name
            
            outputs = session.run(None, {input_name: tensor_img})
            logits = outputs[0]
            
            probabilities = softmax(logits)
            
            # Expected shape: (1, 23)
            return probabilities
        except Exception as e:
            log.error(f"Swin inference failed: {e}")
            raise InferenceError(f"Swin inference failed: {str(e)}")

swin_predictor = SwinPredictor()
