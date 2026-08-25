import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn

from franca.hub.utils import _TEMPDIR, extract_tar_file, load_state_dict_from_url
from rasa.src.rasa_head import RASAHead

_FRANCA_RELEASE_URL = "https://github.com/valeoai/Franca/releases/download/{release}"
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
    IN21K_224 = "In21K_224"
    LAION = "Laion600M"
    LAION_224 = "Laion600M_224"
    DINOV2_IN21K = "Dinov2_In21K"
    DINOV2_IN21K_518 = "Dinov2_In21K_518"
    LAION_518 = "Laion_518"


@dataclass(frozen=True)
class WeightSpec:
    img_size: int
    release: str
    rasa_available: bool = True


_WEIGHT_SPECS = {
    ("vit_base", Weights.IN21K): WeightSpec(img_size=518, release="v1.0.0"),
    ("vit_base", Weights.IN21K_224): WeightSpec(img_size=224, release="v1.1.0", rasa_available=False),
    ("vit_base", Weights.DINOV2_IN21K): WeightSpec(img_size=224, release="v1.1.0", rasa_available=False),
    ("vit_base", Weights.DINOV2_IN21K_518): WeightSpec(img_size=518, release="v1.1.0"),
    ("vit_large", Weights.LAION): WeightSpec(img_size=518, release="v1.0.0"),
    ("vit_large", Weights.LAION_224): WeightSpec(img_size=224, release="v1.1.0", rasa_available=False),
    ("vit_large", Weights.DINOV2_IN21K): WeightSpec(img_size=224, release="v1.1.0", rasa_available=False),
    ("vit_large", Weights.DINOV2_IN21K_518): WeightSpec(img_size=518, release="v1.1.0"),
    ("vit_large", Weights.LAION_518): WeightSpec(img_size=518, release="v1.1.0"),
    ("vit_giant2", Weights.LAION): WeightSpec(img_size=224, release="v1.0.0"),
}


def _normalize_weights(weights: Union[Weights, str]) -> Weights:
    if isinstance(weights, Weights):
        return weights
    try:
        return Weights[weights]
    except KeyError as error:
        supported = ", ".join(weight.name for weight in Weights)
        raise ValueError(f"Unsupported weights: {weights}. Supported values: {supported}") from error


def _get_weight_spec(arch_name: str, weights: Weights) -> WeightSpec:
    try:
        return _WEIGHT_SPECS[(arch_name, weights)]
    except KeyError as error:
        supported = ", ".join(weight.name for arch, weight in _WEIGHT_SPECS if arch == arch_name)
        raise ValueError(f"Weights {weights.name} are not available for {arch_name}. Supported values: {supported}") from error


def _make_checkpoint_url(arch_name: str, patch_size: int, weights: Weights, rasa: bool = False) -> str:
    spec = _get_weight_spec(arch_name, weights)
    base_url = _FRANCA_RELEASE_URL.format(release=spec.release)
    if rasa:
        model_name = _make_rasa_model_name(arch_name, patch_size, weights.value)
    else:
        model_name = _make_franca_model_name(arch_name, patch_size, weights.value)
    return f"{base_url}/{model_name}.pth"


def _normalize_rasa_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("rasa_head."):
            key = key[len("rasa_head.") :]
        normalized[key] = value
    return normalized


def _make_rasa_head(input_dim: int, state_dict: Mapping[str, torch.Tensor]) -> RASAHead:
    state_dict = _normalize_rasa_state_dict(state_dict)
    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("pre_pos_layers.") and key.endswith(".weight")
    )
    if layer_indices != list(range(len(layer_indices))):
        raise ValueError(f"RASA preprocessing layers must be contiguous from zero; found {layer_indices}")
    if "pos_pred.weight" not in state_dict:
        raise ValueError("RASA state dict is missing pos_pred.weight")

    rasa_head = RASAHead(input_dim=input_dim, n_pos_layers=len(layer_indices), pos_out_dim=2)
    rasa_head.load_state_dict(state_dict, strict=True)
    return rasa_head


def _make_franca_model(
    *,
    arch_name: str = "vit_large",
    img_size: Optional[int] = None,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IN21K,
    local_state_dict: Optional[str | list[str]] = None,
    RASA_local_state_dict: Optional[str | list[str]] = None,
    **kwargs,
) -> nn.Module:
    from ..models import build_model

    weights = _normalize_weights(weights)
    weight_spec = _WEIGHT_SPECS.get((arch_name, weights))
    if pretrained and weight_spec is None:
        _get_weight_spec(arch_name, weights)
    if img_size is None:
        img_size = weight_spec.img_size if weight_spec is not None else 224

    # Extract use_rasa_head from kwargs before passing to FrancaConfig
    use_rasa_head = kwargs.pop("use_rasa_head", False)

    vit_config = FrancaConfig(arch=arch_name, use_rasa_head=use_rasa_head, **kwargs)
    model, _ = build_model(vit_config, only_teacher=True, img_size=img_size)

    model_full_name = _make_franca_model_name(arch_name, vit_config.patch_size, weights.value)

    if pretrained:
        if local_state_dict is not None:
            if os.path.isdir(local_state_dict):
                with tempfile.TemporaryDirectory(dir=_TEMPDIR) as tmpdirname:
                    outfile = extract_tar_file(local_state_dict, tmpdirname)
                    state_dict = torch.load(os.path.join(tmpdirname, outfile), map_location="cpu", weights_only=True)
            else:
                state_dict = torch.load(local_state_dict, map_location="cpu", weights_only=True)
        else:
            assert weight_spec is not None
            if arch_name == "vit_giant2":
                base_url = _FRANCA_RELEASE_URL.format(release=weight_spec.release)
                url = [base_url + f"/{model_full_name}_{chunk}" for chunk in _FRANCA_ViT_G_CHUNKS]
            else:
                url = _make_checkpoint_url(arch_name, vit_config.patch_size, weights)
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
        if RASA_local_state_dict is None and (weight_spec is None or not weight_spec.rasa_available):
            raise ValueError(f"RASA weights are not published for {arch_name} with {weights.name} weights.")
        if RASA_local_state_dict is not None:
            rasa_state_dict = torch.load(RASA_local_state_dict, map_location="cpu", weights_only=True)
        else:
            rasa_url = _make_checkpoint_url(arch_name, vit_config.patch_size, weights, rasa=True)
            rasa_state_dict = load_state_dict_from_url(rasa_url, map_location="cpu", weights_only=True)

        model.rasa_head = _make_rasa_head(model.embed_dim, rasa_state_dict)

    if not vit_config.use_rasa_head and hasattr(model, "rasa_head"):
        del model.rasa_head

    return model


def franca_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.IN21K, **kwargs) -> nn.Module:
    """
    Franca ViT-B/14 model (optionally) pretrained on the In21K dataset.
    """
    img_size = kwargs.pop("img_size", None)
    return _make_franca_model(arch_name="vit_base", pretrained=pretrained, weights=weights, img_size=img_size, **kwargs)


def franca_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LAION, **kwargs) -> nn.Module:
    """
    Franca ViT-L/14 model (optionally) pretrained on the LAION-600M dataset by default.
    """
    img_size = kwargs.pop("img_size", None)
    return _make_franca_model(arch_name="vit_large", pretrained=pretrained, weights=weights, img_size=img_size, **kwargs)


def franca_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LAION, **kwargs) -> nn.Module:
    """
    Franca ViT-g/14 model (optionally) pretrained on the LAION-600M dataset by default.
    """
    img_size = kwargs.pop("img_size", None)
    return _make_franca_model(arch_name="vit_giant2", weights=weights, pretrained=pretrained, img_size=img_size, **kwargs)
