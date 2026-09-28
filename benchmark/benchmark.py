
import time

import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor

from app.config import Config
from app.model import ModelFactory
from app.lora import load_lora_state_dict
from optimization.quantize import merge_lora


config = Config()
torch.set_num_threads(4)


def load_fp32_model():
    print("Loading FP32 model...")

    model = ModelFactory(config).create_lora_model(
        config.lora_rank
    )

    checkpoint = torch.load(
        config.checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    load_lora_state_dict(
        model,
        checkpoint["lora_state_dict"]
    )

    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    model.cpu().eval()
    merge_lora(model)

    return model


def load_int8_model():
    print("Loading INT8 model...")

    model = ModelFactory(config).create_lora_model(
        config.lora_rank
    )

    merge_lora(model)
    model.cpu().eval()

    model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    state_dict = torch.load(
        config.quantized_model_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


@torch.inference_mode()
def benchmark(model, inputs, name):
    # Warm up the model before measuring.
    for _ in range(5):
        model(**inputs)

    times = []

    for _ in range(30):
        start = time.perf_counter()
        model(**inputs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    average_seconds = sum(times) / len(times)
    latency_ms = average_seconds * 1000
    throughput = 1 / average_seconds

    print(f"\n{name} RESULTS")
    print(f"Average latency: {latency_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")


def main():
    processor = AutoImageProcessor.from_pretrained(
        config.model_name
    )

    # Identical input for both models.
    image = Image.new("RGB", (224, 224), "green")
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    fp32_model = load_fp32_model()
    benchmark(fp32_model, inputs, "FP32")

    del fp32_model

    int8_model = load_int8_model()
    benchmark(int8_model, inputs, "INT8")


if __name__ == "__main__":
    main()