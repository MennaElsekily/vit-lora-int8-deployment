import torch
from torch import nn
from transformers import AutoImageProcessor
from PIL import Image

from app.config import Config
from app.model import ModelFactory
from optimization.quantize import merge_lora


config = Config()

print("Recreating model architecture...")

# Create the original ViT with LoRA
model = ModelFactory(config).create_lora_model(
    config.lora_rank
)

# Remove the LoRA wrappers to recreate
# the architecture used during quantization.
merge_lora(model)

model.cpu().eval()

# Recreate the quantized architecture
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# Load your saved INT8 weights
print("Loading quantized checkpoint...")

state_dict = torch.load(
    config.quantized_model_path,
    map_location="cpu",
    weights_only=True
)

quantized_model.load_state_dict(
    state_dict,
    strict=True
)

quantized_model.eval()

print("Quantized model loaded successfully!")

# Test inference
processor = AutoImageProcessor.from_pretrained(
    config.model_name
)

image = Image.new("RGB", (224, 224), "green")

inputs = processor(
    images=image,
    return_tensors="pt"
)

with torch.inference_mode():
    outputs = quantized_model(**inputs)
    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

print("Output shape:", outputs.logits.shape)
print("Predicted class:", probabilities.argmax().item())
print("Confidence:", probabilities.max().item())
print("INT8 inference successful!")