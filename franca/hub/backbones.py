import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from franca.hub.utils import _TEMPDIR, extract_tar_file, load_state_dict_from_url
from rasa.src.rasa_head import RASAHead

_FRANCA_BASE_URL = "https://github.com/valeoai/Franca/releases/download/v1.0.0"
_FRANCA_ViT_G_CHUNKS = [
    "chunked.tar.gz.part_aa",
    "chunked.tar.gz.part_ab",
    "chunked.tar.gz.part_ac",
]


def _make_franca_model_name(arch_name: str, patch_size: int, pretraining_dataset: str) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"franca_{compact_arch_name}{patch_size}_{pretraining_dataset}"


def _make_rasa_model_name(arch_name: str, patch_size: int, pretraining_dataset: str) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"franca_{compact_arch_name}{patch_size}_{pretraining_dataset}_rasa"


@dataclass
class FrancaConfig:
    arch: str = "vit_large"
    patch_size: int = 14
    layerscale: float = 1.0e-05
    ffn_layer: str = "swiglufused"
    block_chunks: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1
    use_rasa_head: bool = False


class Weights(Enum):
    IN21K = "In21K"
    LAION = "Laion600M"
    DINOV2_IN21K = "Dinov2_In21K"


def _make_franca_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 224,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IN21K,
    local_state_dict: Optional[str | list[str]] = None,
    RASA_local_state_dict: Optional[str | list[str]] = None,
    **kwargs,
) -> nn.Module:
    from ..models import build_model

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    vit_config = FrancaConfig(arch=arch_name, **kwargs)
    model, _ = build_model(vit_config, only_teacher=True, img_size=img_size)

    model_full_name = _make_franca_model_name(arch_name, vit_config.patch_size, weights.value)

    if pretrained:
        if weights == Weights.LAION and arch_name == "vit_base":
            raise ValueError(
                "Franca ViT-B/14 model is not available with LAION weights. "
                "Please use IN21K weights or set `pretrained=False`."
            )
        if local_state_dict is not None:
            if os.path.isdir(local_state_dict):
                with tempfile.TemporaryDirectory(dir=_TEMPDIR) as tmpdirname:
                    outfile = extract_tar_file(local_state_dict, tmpdirname)
                    state_dict = torch.load(os.path.join(tmpdirname, outfile), map_location="cpu", weights_only=True)
            else:
                state_dict = torch.load(local_state_dict, map_location="cpu", weights_only=True)
        else:
            if arch_name == "vit_giant2":
                url = [_FRANCA_BASE_URL + f"/{model_full_name}_{chunk}" for chunk in _FRANCA_ViT_G_CHUNKS]
            else:
                url = _FRANCA_BASE_URL + f"/{model_full_name}.pth"
            state_dict = load_state_dict_from_url(url, map_location="cpu", weights_only=True)

        state_dict: dict[str, Any] = state_dict["teacher"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        # Filter out rasa_head keys from the main state_dict if they exist
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("rasa_head.")}

        msg = model.load_state_dict(filtered_state_dict, strict=False)

        if len(msg.missing_keys) != 0:
            # Filter out rasa_head keys from missing keys check
            non_rasa_missing = [k for k in msg.missing_keys if not k.startswith("rasa_head.")]
            if len(non_rasa_missing) != 0:
                raise ValueError(
                    f"Missing keys in the state_dict: {non_rasa_missing}. "
                    "Ensure that the model architecture matches the state_dict."
                )

        for k in msg.unexpected_keys:
            if k.startswith("dino_head.") or k.startswith("ibot_head.") or k.startswith("rasa_head."):
                continue
            raise ValueError(
                f"Unexpected key in the state_dict: {k}. Ensure that the model architecture matches the state_dict."
            )

    flat_blocks = []
    for chunk in model.blocks:
        for blk in chunk:
            if not isinstance(blk, nn.Identity):
                flat_blocks.append(blk)
    model.blocks = nn.ModuleList(flat_blocks)
    model.chunked_blocks = False
    assert len(model.blocks) == model.n_blocks, f"Expected {model.n_blocks} blocks, but got {len(model.blocks)} blocks."

    if vit_config.use_rasa_head:
        rasa_head = RASAHead(input_dim=model.embed_dim, n_pos_layers=9, pos_out_dim=2)

        if RASA_local_state_dict is not None:
            rasa_state_dict = torch.load(RASA_local_state_dict, map_location="cpu", weights_only=True)
        else:
            rasa_model_name = _make_rasa_model_name(arch_name, vit_config.patch_size, weights.value)
            rasa_url = _FRANCA_BASE_URL + f"/{rasa_model_name}.pth"
            rasa_state_dict = load_state_dict_from_url(rasa_url, map_location="cpu", weights_only=True)

        # Load the RASA head state dict
        msg = rasa_head.load_state_dict(rasa_state_dict)
        if len(msg.missing_keys) != 0:
            raise ValueError(
                f"Missing keys in the RASA head state_dict: {msg.missing_keys}. "
                "Ensure that the RASA head architecture matches the state_dict."
            )

        model.rasa_head = rasa_head

    if not vit_config.use_rasa_head and hasattr(model, "rasa_head"):
        del model.rasa_head

    return model


def franca_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-B/14 model (optionally) pretrained on the In21K dataset.
    """
    if weights == Weights.DINOV2_IN21K or weights == "Dinov2_In21k":
        img_size = kwargs.pop("img_size", 224)
    else:
        img_size = kwargs.pop("img_size", 518)
    return _make_franca_model(arch_name="vit_base", pretrained=pretrained, weights=weights, img_size=img_size, **kwargs)


def franca_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-L/14 model (optionally) pretrained on either the In21K or Laion600M dataset.
    """
    return _make_franca_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def franca_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-g/14 model (optionally) pretrained on either the In21K or Laion600M dataset.
    """
    return _make_franca_model(arch_name="vit_giant2", weights=weights, pretrained=pretrained, **kwargs)
