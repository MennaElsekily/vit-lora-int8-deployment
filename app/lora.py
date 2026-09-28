import math
import torch
from torch import nn


class ManualLoRALinear(nn.Module):
    """Manual LoRA wrapper for a linear layer."""

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()

        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the original pretrained Linear layer.
        for parameter in self.original.parameters():
            parameter.requires_grad = False

        # LoRA A: rank x input_features
        self.lora_A = nn.Parameter(
            torch.empty(rank, original_layer.in_features)
        )

        # LoRA B: output_features x rank
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, rank)
        )

        # Initialize A and B.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Original frozen projection.
        base_output = self.original(x)

        # LoRA update:
        # x -> A -> B
        lora_output = (x @ self.lora_A.t()) @ self.lora_B.t()

        # Combine pretrained output with LoRA update.
        return base_output + self.scaling * lora_output


def apply_lora(model, rank: int, alpha: int):
    """
    Apply manual LoRA to Query and Value projections
    of the last three Transformer blocks.

    For the current Hugging Face ViT structure:
        model.vit.layers[9]
        model.vit.layers[10]
        model.vit.layers[11]

    LoRA is applied to:
        attention.q_proj
        attention.v_proj
    """

    # Current Transformers ViT structure:
    # model.vit.layers is a ModuleList containing 12 blocks.
    layers = model.vit.layers

    # Last three Transformer blocks: 9, 10, 11
    for block in layers[-3:]:
        attention = block.attention

        # Query projection
        attention.q_proj = ManualLoRALinear(
            attention.q_proj,
            rank=rank,
            alpha=alpha,
        )

        # Value projection
        attention.v_proj = ManualLoRALinear(
            attention.v_proj,
            rank=rank,
            alpha=alpha,
        )

    return model


def freeze_for_lora(model):
    """
    Freeze all pretrained ViT parameters.

    Only the following parameters remain trainable:
        - LoRA A matrices
        - LoRA B matrices
        - Classification head
    """

    # Freeze everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Enable LoRA parameters.
    for module in model.modules():
        if isinstance(module, ManualLoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    # The original ImageNet classifier is replaced/reinitialized
    # for the 10 EuroSAT classes, so it is trainable.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def lora_state_dict(model):
    """
    Return only LoRA A/B parameters.

    Frozen pretrained ViT weights are NOT included.
    """

    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state_dict(model, state_dict):
    """
    Load saved LoRA A/B parameters into a recreated LoRA model.

    The original pretrained ViT weights must already be loaded
    before calling this function.
    """

    named_parameters = dict(model.named_parameters())

    for name, tensor in state_dict.items():

        if name not in named_parameters:
            raise KeyError(
                f"LoRA parameter not found in recreated model: {name}"
            )

        named_parameters[name].data.copy_(
            tensor.to(named_parameters[name].device)
        )