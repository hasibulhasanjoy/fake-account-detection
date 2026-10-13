from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.model_service import model_service
from src.api.schemas import (
    AccountData,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
)
from src.exception import CustomException
from src.logger import logging

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except Exception as e:
        logging.error(f"Failed to load model artifacts on startup: {e}")
        raise RuntimeError(
            "Could not load model artifacts (artifacts/model.pkl and "
            "artifacts/preprocessor.pkl). Run 'python train.py' first."
        ) from e
    yield


app = FastAPI(
    title="Fake Account Detection API",
    description=(
        "Classifies social media accounts as real or fake using profile "
        "metadata, activity counts, and free-text fields. All fields are "
        "optional; missing values are imputed with the statistics learned "
        "during training."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    logging.error(f"CustomException on {request.url.path}: {exc.error_message}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal prediction error. Check server logs."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model_service.model_loaded else "error",
        model_loaded=model_service.model_loaded,
        preprocessor_loaded=model_service.preprocessor_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(account: AccountData):
    return model_service.predict_single(account)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(accounts: list[AccountData]):
    return BatchPredictionResponse(results=model_service.predict_batch(accounts))
