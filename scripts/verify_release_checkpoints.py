#!/usr/bin/env python3
"""Strict-load and optionally forward-test the v1.1.0 release checkpoints."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from franca.hub.backbones import _make_franca_model


WEIGHT_NAMES = {
    "franca_vitb14_in21k_518": "IN21K_518",
    "dinov2_vitb14_in21k_518": "DINOV2_IN21K_518",
    "dinov2_vitl14_in21k_518": "DINOV2_IN21K_518",
    "franca_vitl14_laion_518": "LAION_518",
    "dinov2_vitb14_in21k_224": "DINOV2_IN21K",
    "dinov2_vitl14_in21k_224": "DINOV2_IN21K",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets_dir", type=Path)
    parser.add_argument("--skip-forward", action="store_true", help="Only test checkpoint structure and strict model loading")
    args = parser.parse_args()

    metadata = json.loads((args.assets_dir / "CHECKPOINTS.json").read_text(encoding="utf-8"))
    for item in metadata["models"]:
        use_rasa_head = "rasa_asset" in item
        rasa_path = str(args.assets_dir / item["rasa_asset"]) if use_rasa_head else None
        model = _make_franca_model(
            arch_name=item["arch"],
            weights=WEIGHT_NAMES[item["id"]],
            local_state_dict=str(args.assets_dir / item["backbone_asset"]),
            RASA_local_state_dict=rasa_path,
            use_rasa_head=use_rasa_head,
        )
        assert model.patch_embed.img_size == (item["resolution"], item["resolution"])

        if not args.skip_forward:
            model.eval()
            image = torch.zeros(1, 3, item["resolution"], item["resolution"])
            with torch.inference_mode():
                features = model.forward_features(image, use_rasa_head=use_rasa_head)
            expected_patches = (item["resolution"] // 14) ** 2
            assert features["x_norm_patchtokens"].shape == (1, expected_patches, item["embed_dim"])
            if use_rasa_head:
                assert features["patch_token_rasa"].shape == (1, expected_patches, item["embed_dim"])

        print(f"verified {item['id']}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
