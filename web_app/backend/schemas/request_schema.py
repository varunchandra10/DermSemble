from pydantic import BaseModel, Field
from typing import Optional

class BatchPredictRequest(BaseModel):
    """
    Schema for a batch prediction request.
    Ideally, in a real application, this would contain a list of base64 encoded strings
    or refer to IDs of already uploaded files.
    Here we define a simple structure if base64 images are sent.
    """
    images: list[str] = Field(..., description="List of base64 encoded images.")
