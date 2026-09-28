from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from app.inference import ViTPredictor


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    predictor = ViTPredictor()
    yield
    predictor = None


app = FastAPI(
    title="EuroSAT ViT LoRA INT8 API",
    description="Quantized Vision Transformer image classification",
    version="1.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "EuroSAT ViT API",
        "model": "ViT LoRA 16 INT8"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if predictor is not None else "unavailable",
        "device": "cpu"
    }


@app.post("/predict")
def predict(images: UploadFile = File(...)):
    if images.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    try:
        image = Image.open(
            BytesIO(images.file.read())
        )
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    return predictor.predict(image)