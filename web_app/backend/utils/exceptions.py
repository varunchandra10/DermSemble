class AppError(Exception):
    """Base class for all application specific exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class InvalidImageError(AppError):
    """Exception raised when an image is invalid or cannot be processed."""
    def __init__(self, message: str = "Invalid or corrupted image data."):
        super().__init__(message, status_code=400)

class ModelNotLoadedError(AppError):
    """Exception raised when an inference model is not loaded."""
    def __init__(self, message: str = "Model is not loaded and cannot perform inference."):
        super().__init__(message, status_code=503)

class InferenceError(AppError):
    """Exception raised when an error occurs during model inference."""
    def __init__(self, message: str = "An error occurred during inference."):
        super().__init__(message, status_code=500)

class UnsupportedFormatError(AppError):
    """Exception raised when an uploaded image format is not supported."""
    def __init__(self, message: str = "Unsupported image format. Allowed formats are JPEG and PNG."):
        super().__init__(message, status_code=415)
