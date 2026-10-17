"""Ví dụ đếm TP/FP/FN bằng tay, không chỉ kiểm tra lại chính công thức code."""

import unittest

import torch

from .metrics import binary_metrics, binary_mask_metrics, last_logits


class MetricTests(unittest.TestCase):
    def test_confusion_matrix_hand_calculated(self):
        # TP=2, FP=1, FN=1: Dice=4/6, IoU=2/4, precision=recall=2/3.
        pred = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.bool)
        gt = torch.tensor([[[[1, 0], [1, 1]]]], dtype=torch.bool)
        result = binary_mask_metrics(pred, gt)
        for key, expected in {"dice": 2/3, "iou": 1/2, "precision": 2/3, "recall": 2/3}.items():
            self.assertAlmostEqual(result[key].item(), expected)

    def test_empty_conventions(self):
        # both empty; only GT has positive; only prediction has positive.
        pred = torch.tensor([0, 0, 1], dtype=torch.bool).reshape(3, 1, 1, 1)
        gt = torch.tensor([0, 1, 0], dtype=torch.bool).reshape(3, 1, 1, 1)
        result = binary_mask_metrics(pred, gt)
        self.assertEqual(result["dice"].tolist(), [1, 0, 0])
        self.assertEqual(result["iou"].tolist(), [1, 0, 0])
        self.assertEqual(result["precision"].tolist(), [1, 0, 0])
        self.assertEqual(result["recall"].tolist(), [1, 0, 1])

    def test_equal_image_weighting_differs_from_pixel_pooling(self):
        # Ảnh nhỏ có 1 pixel polyp đoán đúng; ảnh khác có 4 pixel bỏ sót.
        pred = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool).reshape(2,1,2,2)
        gt = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool).reshape(2,1,2,2)
        result = binary_mask_metrics(pred, gt)
        self.assertEqual(result["dice"].mean().item(), 0.5)
        self.assertNotEqual(result["dice"].mean().item(), 2/6)

    def test_threshold_equality_and_one_sigmoid(self):
        logits = torch.tensor([0.0, -0.01]).reshape(1,1,1,2)
        gt = torch.tensor([1.0, 0.0]).reshape(1,1,1,2)
        self.assertEqual(binary_metrics(logits, gt)["dice"].item(), 1)

    def test_shape_mask_and_finite_validation(self):
        valid = torch.zeros(1,1,2,2)
        with self.assertRaises(ValueError):
            binary_metrics(valid, valid + .2)
        with self.assertRaises(ValueError):
            binary_metrics(valid + float("nan"), valid)
        with self.assertRaises(ValueError):
            binary_metrics(valid, valid, threshold=1)
        with self.assertRaises(ValueError):
            binary_metrics(valid, valid[:,0])

    def test_final_head_selection(self):
        a, b = torch.zeros(1), torch.ones(1)
        self.assertIs(last_logits(a), a)
        self.assertIs(last_logits([a,b]), b)
        with self.assertRaises(TypeError):
            last_logits([])


if __name__ == "__main__":
    unittest.main()
