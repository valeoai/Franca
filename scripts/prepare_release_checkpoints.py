#!/usr/bin/env python3
"""Export the four 518px backbone/RASA pairs for the v1.1.0 release."""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch


RELEASES = (
    {
        "id": "franca_vitb14_in21k_518",
        "arch": "vit_base",
        "resolution": 518,
        "dataset": "ImageNet training at 518px, initialized from ImageNet-21K weights",
        "backbone_source": "teachers/franca_vitb/teacher_checkpoint.pth",
        "rasa_source": "franca_vitb14_In21K_rasa.pth",
        "backbone_asset": "franca_vitb14_In21K_518.pth",
        "rasa_asset": "franca_vitb14_In21K_518_rasa.pth",
        "rasa_provenance": "standalone RASA checkpoint",
        "expected_backbone_keys": 175,
        "embed_dim": 768,
    },
    {
        "id": "dinov2_vitb14_in21k_518",
        "arch": "vit_base",
        "resolution": 518,
        "dataset": "DINOv2 In21K baseline, 518px HRFT checkpoint",
        "backbone_source": "teachers/dinov2_vitb_hrft_518/teacher_checkpoint.pth",
        "rasa_source": "teachers/dinov2_vitb_hrft_518/teacher_checkpoint.pth",
        "backbone_asset": "franca_vitb14_Dinov2_In21K_518.pth",
        "rasa_asset": "franca_vitb14_Dinov2_In21K_518_rasa.pth",
        "rasa_provenance": "embedded RASA tensors; training provenance not independently recorded",
        "expected_backbone_keys": 175,
        "embed_dim": 768,
    },
    {
        "id": "dinov2_vitl14_in21k_518",
        "arch": "vit_large",
        "resolution": 518,
        "dataset": "DINOv2 In21K baseline, 518px HRFT checkpoint",
        "backbone_source": "teachers/dinov2_vitl_hrft_518/teacher_checkpoint.pth",
        "rasa_source": "teachers/dinov2_vitl_hrft_518/teacher_checkpoint.pth",
        "backbone_asset": "franca_vitl14_Dinov2_In21K_518.pth",
        "rasa_asset": "franca_vitl14_Dinov2_In21K_518_rasa.pth",
        "rasa_provenance": "embedded RASA tensors; training provenance not independently recorded",
        "expected_backbone_keys": 343,
        "embed_dim": 1024,
    },
    {
        "id": "franca_vitl14_laion_518",
        "arch": "vit_large",
        "resolution": 518,
        "dataset": "LAION-600M, 518px HRFT checkpoint",
        "backbone_source": "teachers/franca_vitl/teacher_checkpoint.pth",
        "rasa_source": "teachers/franca_vitl/teacher_checkpoint.pth",
        "backbone_asset": "franca_vitl14_Laion_518.pth",
        "rasa_asset": "franca_vitl14_Laion_518_rasa.pth",
        "rasa_provenance": "embedded RASA tensors; training provenance not independently recorded",
        "expected_backbone_keys": 343,
        "embed_dim": 1024,
    },
    {
        "id": "dinov2_vitb14_in21k_224",
        "arch": "vit_base",
        "resolution": 224,
        "dataset": "DINOv2 In21K baseline",
        "backbone_source": "franca_vitb14_Dinov2_In21K.pth",
        "backbone_asset": "franca_vitb14_Dinov2_In21K.pth",
        "rasa_provenance": "RASA is not available for this legacy baseline",
        "expected_backbone_keys": 175,
        "embed_dim": 768,
    },
    {
        "id": "dinov2_vitl14_in21k_224",
        "arch": "vit_large",
        "resolution": 224,
        "dataset": "DINOv2 In21K baseline",
        "backbone_source": "franca_vitl14_Dinov2_In21K.pth",
        "backbone_asset": "franca_vitl14_Dinov2_In21K.pth",
        "rasa_provenance": "RASA is not available for this legacy baseline",
        "expected_backbone_keys": 343,
        "embed_dim": 1024,
    },
)

EXCLUDED_PREFIXES = ("dino_head.", "ibot_head.", "rasa_head.")


def strip_training_prefixes(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    if key.startswith("backbone."):
        key = key[len("backbone.") :]
    return key


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def teacher_state(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = load_checkpoint(path)
    if not isinstance(checkpoint, Mapping) or "teacher" not in checkpoint:
        raise ValueError(f"{path} does not contain a 'teacher' state dict")
    return checkpoint["teacher"]


def extract_backbone(path: Path) -> "OrderedDict[str, torch.Tensor]":
    result = OrderedDict()
    for key, value in teacher_state(path).items():
        key = strip_training_prefixes(key)
        if not key.startswith(EXCLUDED_PREFIXES):
            result[key] = value
    return result


def extract_rasa(path: Path) -> "OrderedDict[str, torch.Tensor]":
    checkpoint = load_checkpoint(path)
    state = checkpoint["teacher"] if isinstance(checkpoint, Mapping) and "teacher" in checkpoint else checkpoint
    normalized = OrderedDict((strip_training_prefixes(key), value) for key, value in state.items())
    embedded = OrderedDict(
        (key[len("rasa_head.") :], value) for key, value in normalized.items() if key.startswith("rasa_head.")
    )
    return embedded or normalized


def validate_states(
    spec: Mapping[str, Any], backbone: Mapping[str, torch.Tensor], rasa: Mapping[str, torch.Tensor] | None
) -> None:
    if len(backbone) != spec["expected_backbone_keys"]:
        raise ValueError(f"{spec['id']}: expected {spec['expected_backbone_keys']} backbone keys, found {len(backbone)}")
    if rasa is not None and len(rasa) != 10:
        raise ValueError(f"{spec['id']}: expected 10 RASA keys, found {len(rasa)}")
    if any(key.startswith(EXCLUDED_PREFIXES) for key in backbone):
        raise ValueError(f"{spec['id']}: training-head tensors remain in backbone export")

    pos_embed = backbone.get("pos_embed")
    patch_weight = backbone.get("patch_embed.proj.weight")
    expected_pos_tokens = (spec["resolution"] // 14) ** 2 + 1
    if pos_embed is None or tuple(pos_embed.shape) != (1, expected_pos_tokens, spec["embed_dim"]):
        raise ValueError(f"{spec['id']}: unexpected pos_embed shape")
    if patch_weight is None or tuple(patch_weight.shape) != (spec["embed_dim"], 3, 14, 14):
        raise ValueError(f"{spec['id']}: unexpected patch embedding shape")


def save_checkpoint(value: Any, path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: Iterable[Path], output_path: Path) -> None:
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(paths)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/hnvme/workspace/v115be14-ws/checkpoints/franca"),
        help="Directory containing teachers/ and the standalone Franca-B RASA checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, Any] = {
        "release": "v1.1.0",
        "format": "Franca teacher backbone with optional standalone RASA",
        "models": [],
    }
    asset_paths = []

    for spec in RELEASES:
        backbone_source = args.source_root / spec["backbone_source"]
        rasa_source = args.source_root / spec["rasa_source"] if spec.get("rasa_source") else None
        if not backbone_source.is_file() or (rasa_source is not None and not rasa_source.is_file()):
            raise FileNotFoundError(f"Missing source for {spec['id']}: {backbone_source} or {rasa_source}")

        backbone = extract_backbone(backbone_source)
        rasa = extract_rasa(rasa_source) if rasa_source is not None else None
        validate_states(spec, backbone, rasa)

        backbone_path = args.output_dir / spec["backbone_asset"]
        save_checkpoint({"teacher": backbone}, backbone_path, args.force)
        asset_paths.append(backbone_path)

        rasa_path = args.output_dir / spec["rasa_asset"] if spec.get("rasa_asset") else None
        if rasa_path is not None and rasa is not None:
            save_checkpoint(rasa, rasa_path, args.force)
            asset_paths.append(rasa_path)

        public_spec = {key: value for key, value in spec.items() if not key.endswith("_source")}
        public_spec["backbone_bytes"] = backbone_path.stat().st_size
        if rasa_path is not None:
            public_spec["rasa_bytes"] = rasa_path.stat().st_size
            public_spec["rasa_preprocessing_layers"] = len(
                [key for key in rasa if key.startswith("pre_pos_layers.") and key.endswith(".weight")]
            )
        metadata["models"].append(public_spec)
        print(f"exported {spec['id']}")

    metadata_path = args.output_dir / "CHECKPOINTS.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_checksums((*asset_paths, metadata_path), args.output_dir / "SHA256SUMS")
    print(f"release assets are ready in {args.output_dir}")


if __name__ == "__main__":
    main()
