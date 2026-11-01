from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from backend.schemas.request_schema import BatchPredictRequest
from backend.schemas.response_schema import PredictionResponse, BatchPredictionResponse, HealthResponse, ModelInfoResponse
from backend.services.prediction_service import prediction_service
from backend.inference.model_loader import model_loader
from backend.core.config import settings
from backend.core.logger import log
from backend.utils.exceptions import AppError

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """
    Endpoint for single image prediction.
    """
    if file.size and file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    try:
        # Read the file bytes safely
        if hasattr(file, "file") and hasattr(file.file, "read"):
            image_bytes = file.file.read()
        else:
            image_bytes = file.read()
        
        # Pass to prediction service
        response = prediction_service.process_single_prediction(image_bytes)
        return response
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        log.error(f"Unexpected error in /predict: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictRequest):
    """
    Endpoint for batch predictions from base64 strings.
    """
    try:
        responses = prediction_service.process_batch_prediction(request.images)
        return BatchPredictionResponse(predictions=responses)
    except AppError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        log.error(f"Unexpected error in /batch-predict: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(status="ok", message="Service is healthy")

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Returns information about loaded production models.
    """
    models_loaded = []
    
    if getattr(model_loader, 'convnext_session', None):
        models_loaded.append("ConvNeXt-Tiny (INT8 ONNX)")
    if getattr(model_loader, 'swin_session', None):
        models_loaded.append("Swin-Transformer Tiny (INT8 ONNX)")
    if getattr(model_loader, 'manifest', None):
        models_loaded.append("Champion Ensemble Policy (Soft Voting w=0.42/0.58)")

    return ModelInfoResponse(
        models_loaded=models_loaded,
        versions={"app": settings.APP_VERSION, "framework": "ONNX Runtime"},
        paths={
            "convnext_int8": str(settings.CONVNEXT_INT8_ONNX_PATH),
            "swin_int8": str(settings.SWIN_INT8_ONNX_PATH),
            "manifest": str(settings.DEPLOYMENT_MANIFEST_PATH),
        }
    )

@router.get("/metrics")
async def get_metrics():
    """
    Endpoint for prometheus metrics (placeholder).
    """
    return JSONResponse({"status": "Metrics endpoint placeholder. Implement prometheus client here."})
