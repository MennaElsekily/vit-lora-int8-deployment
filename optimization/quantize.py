import torch
from torch import nn
from pathlib import Path

from app.config import Config
from app.model import ModelFactory
from app.lora import ManualLoRALinear, load_lora_state_dict


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "lora_rank_16_best.pt"
OUTPUT = ROOT / "models" / "quantized_model.pt"


def merge_lora(module):
    """Merge LoRA weights into the original linear layer."""

    for name, child in list(module.named_children()):
        if isinstance(child, ManualLoRALinear):
            original = child.original

            with torch.no_grad():
                delta = (
                    child.lora_B @ child.lora_A
                ) * child.scaling

                original.weight.add_(
                    delta.to(original.weight.dtype)
                )

            setattr(module, name, original)
        else:
            merge_lora(child)


def main():
    print("Loading original ViT and LoRA...")

    config = Config()
    model = ModelFactory(config).create_lora_model(16)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    load_lora_state_dict(
        model,
        checkpoint["lora_state_dict"]
    )

    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True
    )

    model.cpu().eval()

    print("Merging LoRA weights...")
    merge_lora(model)

    assert not any(
        isinstance(m, ManualLoRALinear)
        for m in model.modules()
    )

    print("Applying INT8 quantization...")

    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        quantized.state_dict(),
        OUTPUT
    )

    print("Quantized model saved:", OUTPUT)
    print(
        "File size:",
        round(OUTPUT.stat().st_size / 1024**2, 2),
        "MB"
    )


if __name__ == "__main__":
    main()