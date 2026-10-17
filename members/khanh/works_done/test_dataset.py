"""Kiểm tra dữ liệu bằng ảnh/mask tạo tại thư mục Unicode tạm."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .dataset import KvasirDataset, get_dataloader, get_transforms, validate_manifest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="polyp_dữ_liệu_")
        self.root = Path(self.temp.name)
        image = np.zeros((32, 48, 3), np.uint8)
        image[3:15, 7:22, 2] = 255  # BGR đỏ; chuyển RGB phải ở channel 0.
        mask = np.zeros((32, 48), np.uint8)
        mask[3:15, 7:22] = 255
        cv2.imencode(".png", image)[1].tofile(self.root / "ảnh.png")
        cv2.imencode(".png", mask)[1].tofile(self.root / "mask.png")
        rows = [{"image_id": str(i), "image_path": "ảnh.png", "mask_path": "mask.png", "split": split}
                for i, split in enumerate(("train", "train", "val", "test"))]
        self.frame = pd.DataFrame(rows)
        self.csv = self.root / "manifest.csv"
        self.frame.to_csv(self.csv, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, split="train", **kwargs):
        return KvasirDataset(self.csv, split, self.root, img_size=(32, 48), **kwargs)

    def test_rgb_mask_and_id_unicode(self):
        rgb, mask, image_id = self.dataset("val", return_id=True)[0]
        self.assertEqual(tuple(rgb.shape), (3, 32, 48))
        self.assertEqual(tuple(mask.shape), (1, 32, 48))
        self.assertTrue(torch.equal(rgb[0], mask[0]))
        self.assertTrue(torch.equal(rgb[1], torch.zeros_like(rgb[1])))
        self.assertEqual(image_id, "2")
        self.assertEqual(set(torch.unique(mask).tolist()), {0.0, 1.0})

    def test_epoch_seed_reproduces_augmented_sample(self):
        first, second = self.dataset(seed=42), self.dataset(seed=42)
        for epoch in (1, 3, 1):
            first.set_epoch(epoch)
            second.set_epoch(epoch)
            for index in range(2):
                left, right = first[index], second[index]
                self.assertTrue(torch.equal(left[0], right[0]))
                self.assertTrue(torch.equal(left[1], right[1]))
                self.assertTrue(set(torch.unique(left[1]).tolist()) <= {0.0, 1.0})

    def test_validation_has_no_random_aug(self):
        dataset = self.dataset("val")
        a = dataset[0]
        dataset.set_epoch(100)
        b = dataset[0]
        self.assertTrue(torch.equal(a[0], b[0]))
        with self.assertRaises(ValueError):
            get_transforms("test", augment=True)

    def test_loader_metadata_and_all_samples(self):
        loader = get_dataloader("train", self.csv, self.root, batch_size=2,
                                img_size=(32, 48), seed=42, return_id=True, augment=False)
        rgb, masks, ids = next(iter(loader))
        self.assertEqual(set(ids), {"0", "1"})
        self.assertEqual(rgb.shape[0], 2)
        self.assertEqual(masks.shape[1], 1)
        self.assertIsNotNone(loader.generator)

    def test_bad_manifest_and_corrupt_input_fail(self):
        duplicate = self.frame.copy()
        duplicate.loc[1, "image_id"] = "0"
        with self.assertRaises(ValueError):
            validate_manifest(duplicate)
        with self.assertRaises(ValueError):
            validate_manifest(self.frame.drop(columns="split"))
        (self.root / "ảnh.png").write_bytes(b"not an image")
        with self.assertRaises(FileNotFoundError):
            self.dataset("val")[0]


if __name__ == "__main__":
    unittest.main()
