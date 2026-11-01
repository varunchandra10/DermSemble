import time
import base64
from backend.core.logger import log
from backend.utils.image_utils import bytes_to_pil
from backend.inference.model_loader import model_loader
from backend.inference.preprocessor import preprocessor
from backend.inference.ensemble_predictor import ensemble_predictor
from backend.services.explainability_service import explainability_service
from backend.services.report_service import report_service
from backend.schemas.response_schema import PredictionResponse, ClassProbability


class PredictionService:
    """
    Central orchestration layer for the Dermsemble diagnostic pipeline.
    """
    def process_single_prediction(self, image_bytes: bytes) -> PredictionResponse:
        start_time = time.time()
        
        # 1. Ingest image
        pil_img = bytes_to_pil(image_bytes)
        
        # 2. Preprocess (DullRazor + Shades-of-Gray + 384x384 Standardization)
        tensor_384, clean_rgb_384 = preprocessor.preprocess_image(pil_img)
        
        # 3. Champion Ensemble ONNX Prediction + Acuity Tuning + Abstention
        pred_dict = ensemble_predictor.predict(tensor_384)
        
        pred_idx = pred_dict["predicted_idx"]
        pred_disease = pred_dict["predicted_label"]
        confidence = float(pred_dict["confidence"])
        is_abstained = pred_dict["is_abstained"]
        clinical_tier = pred_dict["clinical_tier"]
        final_probs = pred_dict["final_probs"]
        
        # 4. Rank Class Probabilities (Top-3 and Full Differential)
        sorted_indices = final_probs.argsort()[::-1]
        
        top_3: list[ClassProbability] = []
        all_probs: list[ClassProbability] = []
        
        for idx in sorted_indices:
            label_name = model_loader.get_label(int(idx))
            p_val = float(final_probs[idx])
            cp = ClassProbability(class_name=label_name, probability=p_val)
            all_probs.append(cp)
            if len(top_3) < 3:
                top_3.append(cp)
                
        # 5. Generate Clinical Summary & Risk Triage Report
        report = report_service.generate_clinical_summary(
            top_predictions=top_3,
            is_abstained=is_abstained,
            clinical_tier=clinical_tier
        )
        
        # 6. Generate Grad-CAM Heatmap Overlay
        heatmap_url = explainability_service.generate_heatmap(
            rgb_image_384=clean_rgb_384,
            tensor_384=tensor_384,
            target_class=pred_idx
        )
        
        latency = (time.time() - start_time) * 1000.0
        
        response = PredictionResponse(
            prediction=pred_disease,
            confidence=round(confidence, 4),
            top_3_predictions=top_3,
            class_probabilities=all_probs,
            report=report,
            heatmap_url=heatmap_url,
            latency_ms=round(latency, 2)
        )
        
        log.info(
            f"Prediction completed: disease='{pred_disease}' conf={confidence*100:.1f}% "
            f"tier='{clinical_tier}' abstained={is_abstained} latency={latency:.1f}ms"
        )
        return response

    def process_batch_prediction(self, base64_images: list[str]) -> list[PredictionResponse]:
        responses = []
        for b64_img in base64_images:
            try:
                if "," in b64_img:
                    b64_img = b64_img.split(",")[1]
                image_bytes = base64.b64decode(b64_img)
                res = self.process_single_prediction(image_bytes)
                responses.append(res)
            except Exception as e:
                log.error(f"Error processing image in batch: {e}")
        return responses


prediction_service = PredictionService()
