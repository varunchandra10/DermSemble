import time
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.core.config import settings
from backend.core.logger import log
from backend.api.routes import router
from backend.inference.model_loader import model_loader
from backend.utils.exceptions import AppError
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events.
    """
    # Startup
    log.info(f"Starting up {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Load all models once at startup
    model_loader.load_all()
    
    yield
    
    # Shutdown
    log.info("Shutting down application. Cleaning up resources...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID and Latency Middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as exc:
        log.exception(f"Unhandled exception during request {request_id}")
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    
    # Log request details
    log.info(
        f"req_id={request_id} method={request.method} path={request.url.path} "
        f"status={response.status_code} latency={formatted_process_time}ms"
    )
    
    response.headers["X-Process-Time"] = formatted_process_time
    response.headers["X-Request-ID"] = request_id
    
    return response

# Exception Handlers
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log.warning(f"AppError: {exc.message} (status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )

# Routers
app.include_router(router, prefix=settings.API_PREFIX)

# Mount static files (heatmaps, HTML, etc)
try:
    import os
    os.makedirs("backend/static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="backend/static"), name="static")
except Exception as e:
    log.error(f"Failed to mount static directory: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
