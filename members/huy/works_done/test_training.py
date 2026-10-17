"""Kiểm tra loss và trainer với fixture nhỏ; không phải kết quả Kvasir.

Fixture test không có tệp ảnh thật: trainer phải hoàn thành chỉ với train/val.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import BCESoftDiceLoss, deep_supervision_loss
from .run_utils import PROJECT_ROOT, resolve_config, training_autocast_dtype
from .training import _check_resume_destination, _validate_epoch, train_experiment


class LossTests(unittest.TestCase):
    def test_manual_bce_soft_dice_and_gradients(self):
        logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
        targets = torch.tensor([[[[1., 0.], [1., 0.]]], [[[0., 0.], [0., 0.]]]])
        criterion = BCESoftDiceLoss(smooth=1e-6)
        loss = criterion.per_image(logits, targets)
        expected_dice = torch.tensor([(2 + 1e-6) / (4 + 1e-6), 1e-6 / (2 + 1e-6)])
        expected = .5 * np.log(2) + .5 * (1 - expected_dice)
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.any(logits.grad != 0))

    def test_deep_supervision_equal_mean_and_each_head_gradient(self):
        heads = [torch.full((2, 1, 4, 4), float(index - 2), requires_grad=True) for index in range(4)]
        targets = torch.ones_like(heads[0])
        criterion = BCESoftDiceLoss()
        actual = deep_supervision_loss(heads, targets, criterion)
        expected = sum(criterion(head, targets) for head in heads) / 4
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(head.grad is not None and torch.isfinite(head.grad).all() for head in heads))
        with self.assertRaises(ValueError):
            deep_supervision_loss(heads[:3], targets, criterion)

    def test_invalid_mask_or_nonfinite_is_rejected(self):
        criterion = BCESoftDiceLoss()
        with self.assertRaises(ValueError):
            criterion(torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2), .25))
        with self.assertRaises(ValueError):
            criterion(torch.full((1, 1, 2, 2), float("nan")), torch.zeros(1, 1, 2, 2))

    def test_config_rejects_protocol_mistakes(self):
        for change in ({"num_workers": 2}, {"model": "unet", "model_kwargs": {"deep_supervision": True}},
                       {"image_size": [31, 32]}, {"scheduler": "cosine"}, {"typo": 1},
                       {"amp_dtype": "int8"}, {"amp_dtype": None}):
            with self.assertRaises((ValueError, TypeError)):
                resolve_config(change)

    def test_bf16_rejects_unsupported_gpu_instead_of_changing_precision(self):
        config = resolve_config({"amp": True, "amp_dtype": "bfloat16"})
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "does not support BF16"):
                training_autocast_dtype(config, torch.device("cuda"))
            legacy = resolve_config({"amp": True, "amp_dtype": "float16"})
            self.assertEqual(training_autocast_dtype(legacy, torch.device("cuda")), torch.float16)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "CUDA BF16 is needed for the numerical overflow regression")
    def test_bf16_forward_and_backward_stay_finite_when_fp16_overflows(self):
        # Reproduces the failure mechanism: finite FP32 weights/inputs but
        # convolution activations above FP16's largest finite value (65504).
        layer = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).cuda()
        with torch.no_grad():
            layer.weight.fill_(500.0)
        images = torch.full((2, 1, 8, 8), 128.0, device="cuda")
        masks = torch.zeros_like(images)
        with torch.autocast("cuda", dtype=torch.float16):
            fp16_logits = layer(images)
        self.assertFalse(torch.isfinite(fp16_logits).all())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_logits = layer(images)
            loss = BCESoftDiceLoss()(bf16_logits, masks)
        self.assertTrue(torch.isfinite(bf16_logits).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # BF16 uses no GradScaler.
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.any(layer.weight.grad != 0))


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="huy_fixture_", dir=PROJECT_ROOT)
        self.root = Path(self.temporary.name)
        rows = []
        rng = np.random.default_rng(19)
        for index, split in enumerate(["train"] * 5 + ["val"] * 3 + ["test"]):
            image_path = self.root / f"image_{index}.png"
            mask_path = self.root / f"mask_{index}.png"
            if split != "test":
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[3:10, 4:12] = 255
                image = rng.integers(0, 32, size=(16, 16, 3), dtype=np.uint8)
                image[..., 2][mask > 0] = 220
                for path, pixels in ((image_path, image), (mask_path, mask)):
                    success, encoded = cv2.imencode(".png", pixels)
                    if not success:
                        raise RuntimeError("Cannot encode fixture")
                    encoded.tofile(str(path))
            rows.append({"image_id": f"fixture_{index}",
                         "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                         "mask_path": mask_path.relative_to(PROJECT_ROOT).as_posix(), "split": split})
        self.manifest = self.root / "splits.csv"
        pd.DataFrame(rows).to_csv(self.manifest, index=False)
        self.config = {"manifest": str(self.manifest), "image_size": [16, 16],
                       "model_kwargs": {"base_channels": 2}, "batch_size": 2,
                       "max_epochs": 3, "early_stopping_patience": 10,
                       "device": "cpu", "amp": False, "torch_threads": 2,
                       "augmentation": True}

    def tearDown(self):
        self.temporary.cleanup()

    def test_train_checkpoint_and_resume_match_uninterrupted(self):
        uninterrupted = self.root / "uninterrupted"
        staged = self.root / "staged"
        summary = train_experiment(self.config, uninterrupted)
        train_experiment({**self.config, "max_epochs": 1}, staged)
        resumed = train_experiment(self.config, staged, staged / "last.pt")
        original = torch.load(uninterrupted / "last.pt", weights_only=True)
        restored = torch.load(staged / "last.pt", weights_only=True)
        self.assertEqual(summary["test_images_read"], 0)
        self.assertEqual(resumed["epochs_completed"], 3)
        self.assertEqual(restored["split_counts"], {"train": 5, "val": 3, "test": 1})
        for name, parameter in original["model_state"].items():
            torch.testing.assert_close(parameter, restored["model_state"][name], rtol=0, atol=0)
        for left, right in zip(original["history"], restored["history"]):
            for name in ("train_loss", "val_loss", "val_dice", "val_iou"):
                self.assertEqual(left[name], right[name])
        self.assertEqual(original["rng_state"]["numpy"], restored["rng_state"]["numpy"])
        self.assertEqual(len(pd.read_csv(staged / "history.csv")), 3)
        self.assertTrue((staged / "best.pt").is_file())
        self.assertIn("source_hashes", restored["provenance"])
        with self.assertRaises(FileExistsError):
            train_experiment(self.config, staged)
        with self.assertRaisesRegex(ValueError, "experimental protocol"):
            train_experiment({**self.config, "learning_rate": .002}, staged, staged / "last.pt")

    def test_deep_supervision_actual_model_trains_and_reloads(self):
        config = {**self.config, "model": "unetplusplus", "max_epochs": 1,
                  "model_kwargs": {"base_channels": 2, "deep_supervision": True}}
        output = self.root / "deep_supervision"
        result = train_experiment(config, output)
        self.assertTrue(np.isfinite(result["best"]["val_loss"]))
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertTrue(checkpoint["model_kwargs"]["deep_supervision"])
        self.assertEqual(checkpoint["test_images_read"] if "test_images_read" in checkpoint else result["test_images_read"], 0)

    def test_early_stopping_counts_nonimproving_epochs(self):
        validation = {"loss": .6, "dice": .5, "iou": .4, "precision": .5, "recall": .5}
        with patch("members.huy.works_done.training._validate_epoch", return_value=validation):
            result = train_experiment({**self.config, "max_epochs": 8, "early_stopping_patience": 2},
                                      self.root / "early_stopping")
        self.assertEqual(result["status"], "early_stopped")
        self.assertEqual(result["epochs_completed"], 3)
        self.assertEqual(result["best"]["epoch"], 1)

    def test_validation_averages_images_including_short_final_batch(self):
        logits = torch.tensor([10., -10., -10.]).reshape(3, 1, 1, 1).expand(3, 1, 2, 2)
        targets = torch.tensor([1., 0., 1.]).reshape(3, 1, 1, 1).expand(3, 1, 2, 2)
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        criterion = BCESoftDiceLoss()
        result = _validate_epoch(torch.nn.Identity(), loader, criterion, torch.device("cpu"), False, .5)
        self.assertAlmostEqual(result["dice"], 2 / 3)
        self.assertAlmostEqual(result["iou"], 2 / 3)
        self.assertAlmostEqual(result["loss"], criterion.per_image(logits, targets).double().mean().item())

    def test_in_place_resume_rejects_rewind_without_overwriting_history(self):
        run = self.root / "existing_run"
        run.mkdir()
        torch.save({"epoch": 5}, run / "last.pt")
        history = run / "history.csv"
        history.write_text("epoch\n1\n2\n3\n4\n5\n", encoding="utf-8")
        previous_bytes = history.read_bytes()
        with self.assertRaisesRegex(ValueError, "Cannot rewind"):
            _check_resume_destination({"epoch": 2}, run / "best.pt", run)
        self.assertEqual(history.read_bytes(), previous_bytes)
        self.assertIsNone(_check_resume_destination({"epoch": 5}, run / "last.pt", run))

    def test_fork_resume_requires_matching_original_best(self):
        source = self.root / "source_run"
        source.mkdir()
        config = resolve_config(self.config)
        checkpoint = {"epoch": 5, "best": {"epoch": 3, "val_dice": .8, "val_iou": .7, "val_loss": .3},
                      "model_name": config["model"], "model_kwargs": config["model_kwargs"],
                      "config": config, "manifest_sha256": "original-split",
                      "split_counts": {"train": 5, "val": 3, "test": 1},
                      "provenance": {"source_hashes": {"training.py": "same-code"}}}
        best = {**checkpoint, "epoch": 3}
        torch.save(best, source / "best.pt")
        destination = self.root / "forked_run"
        self.assertEqual(_check_resume_destination(checkpoint, source / "last.pt", destination), source / "best.pt")
        for alteration in ({"epoch": 6}, {"manifest_sha256": "another-split"},
                           {"config": {**config, "learning_rate": .1}}):
            torch.save({**best, **alteration}, source / "best.pt")
            with self.assertRaises(ValueError):
                _check_resume_destination(checkpoint, source / "last.pt", destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
