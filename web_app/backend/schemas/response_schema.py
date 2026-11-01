from pydantic import BaseModel, Field

class ClassProbability(BaseModel):
    class_name: str
    probability: float

class PredictionResponse(BaseModel):
    """
    Structured response schema for a single image prediction.
    """
    prediction: str = Field(..., description="The most likely disease prediction.")
    confidence: float = Field(..., description="Confidence score of the prediction (0.0 to 1.0).")
    top_3_predictions: list[ClassProbability] = Field(..., description="Top 3 predictions with probabilities.")
    class_probabilities: list[ClassProbability] = Field(..., description="Probabilities for all classes.")
    report: str = Field(..., description="Clinical summary and risk note.")
    heatmap_url: str | None = Field(None, description="URL or path to the generated GradCAM heatmap.")
    disclaimer: str = Field(
        "AI preliminary diagnosis only. Please consult a dermatologist for medical advice.",
        description="Medical disclaimer."
    )
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")

class BatchPredictionResponse(BaseModel):
    """
    Structured response schema for a batch of predictions.
    """
    predictions: list[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    message: str

class ModelInfoResponse(BaseModel):
    models_loaded: list[str]
    versions: dict[str, str]
    paths: dict[str, str]
