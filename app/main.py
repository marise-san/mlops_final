import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import PROJECT_NAME, VERSION
from app.model import ModelHandler
from app.schemas import BankMarketingInput, PredictionOutput

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Singleton instantiation
model_handler = ModelHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FastAPI. Loading model...")
    model_handler.load_model()
    if not model_handler.is_loaded:
        logger.warning("Startup completed but the model could not be loaded!")
    yield
    logger.info("Shutting down the application.")

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handles incoming request schema validation 422 errors specifically to show the details"""
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input formatting or unrecognized fields.", "errors": exc.errors()}
    )

@app.get("/health")
def check_health():
    """Healthcheck endpoint asserting system state and API info"""
    return {
        "status": "online",
        "api_version": VERSION,
        "model_loaded": model_handler.is_loaded
    }

@app.get("/info")
def get_info():
    """Returns training metrics embedded alongside the model artifact."""
    if not model_handler.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    try:
        metrics = model_handler.get_metadata()
        return {"model_type": "RandomForestClassifier Pipeline", "metrics": metrics}
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        raise HTTPException(status_code=500, detail="Internal server error getting metadata.")

@app.post("/predict", response_model=PredictionOutput)
def perform_prediction(input_data: BankMarketingInput):
    """Performs inference via the pre-processed ML pipeline."""
    if not model_handler.is_loaded:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")

    try:
        resultado = model_handler.predict(input_data)
        return PredictionOutput(**resultado)
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
