import numpy as np
from backend.inference.model_loader import model_loader
from backend.utils.exceptions import InferenceError
from backend.core.logger import log

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

class CNNPredictor:
    """
    Handles inference using the EfficientNet ONNX model.
    """
    def predict(self, tensor_img: np.ndarray) -> np.ndarray:
        try:
            session = model_loader.get_cnn_session()
            
            # Usually ONNX models expect float32
            input_name = session.get_inputs()[0].name
            
            # Run inference
            outputs = session.run(None, {input_name: tensor_img})
            logits = outputs[0]
            
            # Apply softmax to get probabilities
            probabilities = softmax(logits)
            
            # Expected shape: (1, 23)
            return probabilities
        except Exception as e:
            log.error(f"CNN inference failed: {e}")
            raise InferenceError(f"CNN inference failed: {str(e)}")

cnn_predictor = CNNPredictor()
