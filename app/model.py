from transformers import ViTForImageClassification

from .lora import apply_lora, freeze_for_lora

class ModelFactory:
    """Creates a fresh model for each experiment."""

    def __init__(self, config):
        self.config = config

    def create_full_finetuning_model(self):
        model = ViTForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            ignore_mismatched_sizes=True,
        )
        return model

    def create_lora_model(self, rank):
        model = ViTForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            ignore_mismatched_sizes=True,
        )

        model = apply_lora(
            model,
            rank=rank,
            alpha=self.config.lora_alpha,
        )
        freeze_for_lora(model)
        return model
