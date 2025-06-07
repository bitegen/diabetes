import sys
from pathlib import Path
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib

# Приводим корень проекта в sys.path, чтобы “import diabetes_model…” работал
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diabetes_model.config.core import config

app = FastAPI(
    title="Diabetes Prediction API",
    version=config.app_config.version,
    docs_url=None, redoc_url=None
)

HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

MODEL_PATH = (
    ROOT
    / "diabetes_model"
    / "trained_models"
    / f"{config.app_config.pipeline_save_file}{config.app_config.version}.pkl"
)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    # SkinThickness и DiabetesPedigreeFunction мы сняли из UI, 
    # но если модель их не ждёт, их можно вообще убрать из pydantic-модели.
    Insulin: float
    BMI: float
    Age: float


class DiabetesOutput(BaseModel):
    prediction: int = Field(...)


@app.get("/")
def serve_frontend():
    return FileResponse(HERE / "static" / "index.html")


@app.post("/predict", response_model=DiabetesOutput)
def predict(payload: DiabetesInput):
    # В Pydantic v2 вместо .dict() используем .model_dump()
    data = payload.model_dump()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}


if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
