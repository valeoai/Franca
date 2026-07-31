import unittest

import torch

from franca.hub.backbones import Weights, _get_weight_spec, _make_checkpoint_url, _make_rasa_head


class ReleaseRoutingTest(unittest.TestCase):
    def test_518_release_urls(self):
        expected = {
            ("vit_base", Weights.IN21K_518): "franca_vitb14_In21K_518",
            ("vit_base", Weights.DINOV2_IN21K_518): "franca_vitb14_Dinov2_In21K_518",
            ("vit_large", Weights.DINOV2_IN21K_518): "franca_vitl14_Dinov2_In21K_518",
            ("vit_large", Weights.LAION_518): "franca_vitl14_Laion_518",
        }
        for (arch, weights), asset_stem in expected.items():
            with self.subTest(arch=arch, weights=weights):
                self.assertEqual(_get_weight_spec(arch, weights).img_size, 518)
                self.assertEqual(
                    _make_checkpoint_url(arch, 14, weights),
                    f"https://github.com/valeoai/Franca/releases/download/v1.1.0/{asset_stem}.pth",
                )
                self.assertEqual(
                    _make_checkpoint_url(arch, 14, weights, rasa=True),
                    f"https://github.com/valeoai/Franca/releases/download/v1.1.0/{asset_stem}_rasa.pth",
                )

    def test_legacy_weights_stay_on_v1_0_0(self):
        url = _make_checkpoint_url("vit_base", 14, Weights.IN21K)
        self.assertIn("/v1.0.0/", url)
        self.assertEqual(_get_weight_spec("vit_large", Weights.LAION).img_size, 518)

    def test_legacy_dinov2_weights_are_republished(self):
        for arch in ("vit_base", "vit_large"):
            with self.subTest(arch=arch):
                url = _make_checkpoint_url(arch, 14, Weights.DINOV2_IN21K)
                self.assertIn("/v1.1.0/", url)

    def test_unsupported_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not available"):
            _get_weight_spec("vit_base", Weights.LAION_518)

    def test_unavailable_legacy_models_are_rejected(self):
        for arch in ("vit_large", "vit_giant2"):
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "not available"):
                _get_weight_spec(arch, Weights.IN21K)

    def test_rasa_layer_count_is_inferred(self):
        for n_layers in (8, 9):
            state_dict = {"pos_pred.weight": torch.zeros(2, 16)}
            state_dict.update({f"pre_pos_layers.{index}.weight": torch.zeros(2, 16) for index in range(n_layers)})
            with self.subTest(n_layers=n_layers):
                head = _make_rasa_head(16, state_dict)
                self.assertEqual(head.n_pos_layers, n_layers)


if __name__ == "__main__":
    unittest.main()
