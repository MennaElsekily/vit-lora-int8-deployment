
import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor

from app.config import Config
from app.model import ModelFactory


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class ViTPredictor:
    def __init__(self):
        config = Config()

        print("Loading quantized ViT...")

        # 1. Create the original ViT architecture with 10 classes.
        # No LoRA layers are needed because they were already merged
        # before the quantized checkpoint was saved.
        model = ModelFactory(config).create_full_finetuning_model()

        # 2. Prepare the model for CPU inference.
        model.cpu().eval()

        # 3. Reconstruct the INT8 architecture so that it matches
        # the saved quantized checkpoint.
        self.model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

        # 4. Load the trained INT8 weights.
        state_dict = torch.load(
            config.quantized_model_path,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.eval()

        # 5. Load the preprocessing configuration for ViT.
        self.processor = AutoImageProcessor.from_pretrained(
            config.model_name
        )

        print("Quantized ViT ready!")

    @torch.inference_mode()
    def predict(self, image: Image.Image):

        # 1. Convert the uploaded image to RGB.
        image = image.convert("RGB")

        # 2. Resize, normalize, and convert to PyTorch tensors.
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # 3. Run inference using the quantized ViT.
        outputs = self.model(**inputs)

        # 4. Convert raw logits into probabilities.
        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        # 5. Find the class with the highest probability.
        confidence, class_id = probabilities.max(dim=-1)

        predicted_id = int(class_id.item())

        # 6. Return the prediction.
        return {
            "class_id": predicted_id,
            "confidence": float(confidence.item())
        }