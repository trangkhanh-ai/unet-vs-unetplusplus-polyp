"""CPU checks for actual dense connections, logits and supervision gradients."""

from __future__ import annotations

import io
import unittest

import torch
from torch import nn

from .unetplusplus import NestedSkipBlock, UNetPlusPlus


class UNetPlusPlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_default_rgb_output_shape(self):
        model = UNetPlusPlus().eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 256, 256))
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(tuple(logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(logits).all())

    def test_custom_channels_rectangular_and_four_heads(self):
        model = UNetPlusPlus(in_channels=1, out_channels=2, base_channels=2,
                             deep_supervision=True).eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 1, 32, 48))
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(tuple(out.shape) == (2, 2, 32, 48) for out in outputs))

    def test_each_nested_node_receives_every_dense_skip(self):
        b = 2
        model = UNetPlusPlus(base_channels=b).eval()
        observed = {}
        hooks = []
        for i in range(4):
            for j in range(1, 5 - i):
                name = f"conv{i}_{j}"
                block = getattr(model, name)
                self.assertIsInstance(block, NestedSkipBlock)
                channels = b * 2 ** i
                self.assertEqual(block.conv[0].in_channels, (j + 1) * channels)
                self.assertEqual(block.up.in_channels, 2 * channels)
                self.assertEqual(block.up.out_channels, channels)
                # Hook the actual concatenated input, not only declared widths.
                def capture(_, args, key=name):
                    observed[key] = tuple(args[0].shape)
                hooks.append(block.conv.register_forward_pre_hook(capture))
        try:
            with torch.no_grad():
                model(torch.randn(1, 3, 32, 48))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(observed), 10)
        for i in range(4):
            for j in range(1, 5 - i):
                self.assertEqual(observed[f"conv{i}_{j}"],
                                 (1, (j + 1) * b * 2 ** i, 32 // 2 ** i, 48 // 2 ** i))

    def test_plain_output_backpropagates_through_all_nodes(self):
        torch.manual_seed(42)
        model = UNetPlusPlus(base_channels=2)
        logits = model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 2, logits.shape).float()
        nn.BCEWithLogitsLoss()(logits, target).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_deep_supervision_updates_all_four_heads(self):
        torch.manual_seed(42)
        model = UNetPlusPlus(base_channels=2, deep_supervision=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [head.bias.detach().clone() for head in model.heads]
        outputs = model(torch.randn(2, 3, 32, 32))
        target = torch.zeros_like(outputs[-1])
        loss = torch.stack([nn.BCEWithLogitsLoss()(logits, target) for logits in outputs]).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for head in model.heads:
            self.assertGreater(float(head.bias.grad.abs().sum()), 0)
        optimizer.step()
        self.assertTrue(all(not torch.equal(old, head.bias) for old, head in zip(before, model.heads)))

    def test_state_dict_roundtrip_with_batch_norm_and_deep_supervision(self):
        model = UNetPlusPlus(base_channels=2, batch_norm=True, deep_supervision=True).eval()
        x = torch.randn(2, 3, 32, 48)
        buffer = io.BytesIO()
        torch.save({"configuration": model.configuration(), "state_dict": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = UNetPlusPlus(**checkpoint["configuration"]).eval()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        with torch.no_grad():
            before, after = model(x), restored(x)
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_output_is_raw_logits(self):
        # Set a negative constant head: a sigmoid inside the model would fail.
        model = UNetPlusPlus(base_channels=2).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(-2.0)
            output = model(torch.randn(1, 3, 32, 32))
        torch.testing.assert_close(output, torch.full_like(output, -2.0), rtol=0, atol=0)

    def test_b_and_c_start_with_identical_trunk_and_final_head(self):
        # Isolate the DS change: its auxiliary heads must not alter the RNG
        # initialization of the trunk or final prediction head.
        torch.manual_seed(42)
        model_b = UNetPlusPlus(base_channels=2).eval()
        torch.manual_seed(42)
        model_c = UNetPlusPlus(base_channels=2, deep_supervision=True).eval()
        state_c = model_c.state_dict()
        for name, parameter in model_b.state_dict().items():
            counterpart = "heads.3." + name[len("head."):] if name.startswith("head.") else name
            torch.testing.assert_close(parameter, state_c[counterpart], rtol=0, atol=0)
        x = torch.randn(2, 3, 32, 48)
        with torch.no_grad():
            torch.testing.assert_close(model_b(x), model_c(x)[-1], rtol=0, atol=0)

    def test_invalid_configuration(self):
        for kwargs in ({"in_channels": 0}, {"out_channels": -1}, {"base_channels": True},
                       {"base_channels": 2.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                UNetPlusPlus(**kwargs)
        for kwargs in ({"batch_norm": 1}, {"deep_supervision": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                UNetPlusPlus(**kwargs)

    def test_invalid_inputs(self):
        model = UNetPlusPlus(base_channels=2)
        for x in (torch.empty(3, 32, 32), torch.empty(0, 3, 32, 32),
                  torch.empty(1, 1, 32, 32), torch.empty(1, 3, 8, 32),
                  torch.empty(1, 3, 33, 32)):
            with self.subTest(shape=tuple(x.shape)), self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 3, 32, 32, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
