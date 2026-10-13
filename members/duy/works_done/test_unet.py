"""Contract tests: python -m unittest members.duy.works_done.test_unet -v."""

import io
import unittest
from unittest.mock import patch

import torch

from .model_factory import build_model
from .unet import UNet
from .verification_utils import diagnostic_loss, diagnostic_scores, seed_everything


class UNetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42, threads=2)

    def test_default_rgb_contract_at_256(self):
        model = UNet().eval()
        with torch.no_grad():
            logits = model(torch.rand(2, 3, 256, 256))
        self.assertEqual(tuple(logits.shape), (2, 1, 256, 256))
        self.assertTrue(torch.isfinite(logits).all())

    def test_rectangular_spatial_size_and_custom_channels(self):
        model = UNet(in_channels=1, out_channels=2, base_channels=4)
        self.assertEqual(tuple(model(torch.rand(1, 1, 32, 48)).shape), (1, 2, 32, 48))

    def test_optimizer_reaches_all_encoder_and_decoder_parameters(self):
        model = UNet(base_channels=4)
        x, target = torch.rand(2, 3, 32, 32), (torch.rand(2, 1, 32, 32) > 0.5).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.head.bias.detach().clone()
        loss = diagnostic_loss(model(x), target)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head.bias))

    def test_save_load_preserves_predictions_with_batchnorm(self):
        model = UNet(base_channels=4, batch_norm=True)
        model(torch.rand(2, 3, 32, 32))  # Update running statistics before saving.
        model.eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            expected = model(x)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = UNet(**model.configuration()).eval()
        restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_invalid_inputs_fail_with_actionable_errors(self):
        model = UNet(base_channels=4)
        for shape in [(3, 32, 32), (1, 1, 32, 32), (1, 3, 31, 32), (1, 3, 8, 8), (0, 3, 32, 32)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.rand(shape))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 3, 32, 32, dtype=torch.uint8))
        for kwargs in [{"base_channels": 0}, {"base_channels": True}, {"batch_norm": "yes"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                UNet(**kwargs)

    def test_factory_rejects_wrong_name_and_deep_supervision(self):
        self.assertIsInstance(build_model("unet", base_channels=4), UNet)
        with self.assertRaises(ValueError):
            build_model("resnet")
        with self.assertRaises(ValueError):
            build_model("unet", deep_supervision=True)

    def test_missing_phuc_module_is_explicit(self):
        missing = ModuleNotFoundError("not delivered", name="members.phuc")
        with patch("members.duy.works_done.model_factory.import_module", side_effect=missing):
            with self.assertRaisesRegex(ModuleNotFoundError, "not delivered yet"):
                build_model("unetplusplus")

    def test_factory_preserves_errors_inside_phuc_module(self):
        missing = ModuleNotFoundError("missing library", name="a_dependency_of_phuc")
        with patch("members.duy.works_done.model_factory.import_module", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError) as caught:
                build_model("unetplusplus")
        self.assertEqual(caught.exception.name, "a_dependency_of_phuc")

    def test_metric_reference_matches_hand_calculation(self):
        # Pred {0,1}; truth {1,2}: TP=1, FP=1, FN=1 -> Dice .5 / IoU 1/3.
        logits = torch.tensor([[[[10., 10.], [-10., -10.]]]])
        masks = torch.tensor([[[[0., 1.], [1., 0.]]]])
        scores = diagnostic_scores(logits, masks)
        self.assertAlmostEqual(scores["mean_dice"], 0.5)
        self.assertAlmostEqual(scores["mean_iou"], 1 / 3, places=6)
        empty = torch.zeros_like(masks)
        self.assertEqual(diagnostic_scores(torch.full_like(logits, -10), empty)["mean_dice"], 1)
        self.assertEqual(diagnostic_scores(torch.full_like(logits, 10), empty)["mean_dice"], 0)


if __name__ == "__main__":
    unittest.main()
