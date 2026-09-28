
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # =====================================
    # Model configuration
    # =====================================
    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = 10

    # =====================================
    # LoRA configuration
    # =====================================
    lora_rank: int = 16
    lora_alpha: int = 16

    # =====================================
    # Quantization
    # =====================================
    quantization_dtype: str = "int8"

    # =====================================
    # Model paths
    # =====================================
    base_dir = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        base_dir / "models" / "lora_rank_16_best.pt"
    )

    quantized_model_path = (
        base_dir / "models" / "quantized_model.pt"
    )

    # =====================================
    # Inference
    # =====================================
    device: str = "cpu"

    # =====================================
    # FastAPI
    # =====================================
    host: str = "0.0.0.0"
    port: int = 8000