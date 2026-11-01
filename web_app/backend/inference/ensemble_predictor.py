import numpy as np
from backend.inference.model_loader import model_loader
from backend.utils.exceptions import InferenceError
from backend.core.logger import log


class EnsemblePredictor:
    """
    Production ONNX Inference & Champion Ensemble Engine.
    Executes:
      1. Dual INT8 ONNX forward passes (ConvNeXt-Tiny & Swin-Transformer Tiny).
      2. Optimal Weighted Soft Voting: P_ens = 0.42 * P_conv + 0.58 * P_swin.
      3. Acuity-Aware Oncology Safety Multiplier: Alpha = 1.85 for Malignant Melanoma.
      4. Clinical Uncertainty Abstention: Flags cases where max confidence < 0.50.
    """
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def predict(self, input_tensor: np.ndarray) -> dict:
        """
        Runs dual ONNX inference on preprocessed tensor (1, 3, 384, 384).
        
        Returns:
            dict containing:
              - predicted_idx (int)
              - predicted_label (str)
              - confidence (float)
              - is_abstained (bool)
              - clinical_tier (str)
              - final_probs (np.ndarray of shape (25,))
              - conv_probs (np.ndarray of shape (25,))
              - swin_probs (np.ndarray of shape (25,))
        """
        try:
            conv_sess = model_loader.get_convnext_session()
            swin_sess = model_loader.get_swin_session()
            
            # 1. Dual ONNX Inferences
            conv_logits = conv_sess.run(None, {"input_tensor": input_tensor})[0]
            swin_logits = swin_sess.run(None, {"input_tensor": input_tensor})[0]
            
            # 2. Probability Conversion
            p_conv = self._softmax(conv_logits)[0]
            p_swin = self._softmax(swin_logits)[0]
            
            # 3. Champion Weighted Soft Voting (0.42 ConvNeXt + 0.58 Swin-T)
            w_c = model_loader.w_conv
            w_s = model_loader.w_swin
            p_ens = (w_c * p_conv) + (w_s * p_swin)
            
            # 4. Oncology Acuity Safety Multiplier (Alpha = 1.85 for Melanoma)
            mel_idx = model_loader.get_class_id("Malignant_Melanoma")
            alpha_mel = model_loader.alpha_mel
            
            p_tuned = p_ens.copy()
            if mel_idx >= 0 and mel_idx < len(p_tuned):
                p_tuned[mel_idx] *= alpha_mel
                # Re-normalize to ensure valid probability simplex
                p_tuned = p_tuned / np.sum(p_tuned)
                
            # 5. Top Diagnostic Selection & Clinical Abstention Audit
            pred_idx = int(np.argmax(p_tuned))
            confidence = float(p_tuned[pred_idx])
            pred_label = model_loader.get_label(pred_idx)
            tier = model_loader.get_tier(pred_label)
            
            # Abstention threshold theta = 0.50
            theta = model_loader.theta_abstain
            is_abstained = bool(confidence < theta)
            
            return {
                "predicted_idx": pred_idx,
                "predicted_label": pred_label,
                "confidence": confidence,
                "is_abstained": is_abstained,
                "clinical_tier": tier,
                "final_probs": p_tuned,
                "conv_probs": p_conv,
                "swin_probs": p_swin
            }

        except Exception as e:
            log.error(f"Production ensemble prediction failed: {e}")
            raise InferenceError(f"Production ensemble prediction failed: {str(e)}")


ensemble_predictor = EnsemblePredictor()
