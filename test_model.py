from pathlib import Path

import torch
from transformers import AutoImageProcessor

from app.config import Config
from app.model import ModelFactory
from app.lora import load_lora_state_dict


# 1. Configuration
config = Config()
device = "cpu"

# 2. Locate your checkpoint
BASE_DIR = Path(__file__).resolve().parent
checkpoint_path = (
    BASE_DIR / "models" / "lora_rank_16_best.pt"
)

# 3. Recreate your original model
print("Creating ViT with LoRA rank 16...")

factory = ModelFactory(config)
model = factory.create_lora_model(rank=16)

# 4. Load your checkpoint
print("Loading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

# 5. Restore LoRA weights
expected_lora = {
    name for name, _ in model.named_parameters()
    if name.endswith(("lora_A", "lora_B"))
}

saved_lora = set(checkpoint["lora_state_dict"])

assert expected_lora == saved_lora, (
    f"LoRA key mismatch: "
    f"missing={expected_lora - saved_lora}, "
    f"unexpected={saved_lora - expected_lora}"
)

load_lora_state_dict(
    model,
    checkpoint["lora_state_dict"]
)

# 6. Restore classifier
model.classifier.load_state_dict(
    checkpoint["classifier_state_dict"],
    strict=True
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print(
    "Saved validation accuracy:",
    checkpoint["validation_accuracy"]
)

# 7. Test a forward pass
processor = AutoImageProcessor.from_pretrained(
    config.model_name
)

from PIL import Image

test_image = Image.new(
    "RGB", (224, 224), color="green"
)

inputs = processor(
    images=test_image,
    return_tensors="pt"
)

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.softmax(
        outputs.logits, dim=-1
    )

print("Output shape:", outputs.logits.shape)
print("Predicted class:", probabilities.argmax().item())
print("Forward pass successful!")