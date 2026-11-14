"""Kiểm tra các rủi ro đã tìm thấy trong bản gộp trước khi nộp."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from members.huy.works_done.extended_training import (
    resolve_extended_config, train_extended, _check_resume, learning_rate_at,
)
from members.huy.works_done.run_utils import DEFAULT_CONFIG
from members.tri.works_done.demo import append_extended_runs
from members.tri.works_done.submission_analysis import paired_statistics


class SubmissionChecks(unittest.TestCase):
    def test_resume_requires_existing_last(self):
        with self.assertRaises(FileNotFoundError):
            train_extended(DEFAULT_CONFIG, "missing_resume_submission_test", resume=True)

    def test_official_runs_are_protected(self):
        with self.assertRaises(ValueError):
            train_extended(DEFAULT_CONFIG, "A_unet_seed42")

    def test_name_cannot_escape_runs(self):
        with self.assertRaises(ValueError):
            train_extended(DEFAULT_CONFIG, "../overwritten")

    def state(self):
        cfg = resolve_extended_config(DEFAULT_CONFIG)
        state = dict(trainer_kind="extension_20261111", config=copy.deepcopy(cfg),
                     rng_state={}, optimizer_state={}, scaler_state={}, history=[], best={},
                     patience_count=0, epoch=0, provenance={"source_hashes": {"file": "same"}},
                     manifest_sha256="same")
        return cfg, state

    def test_resume_rejects_learning_rate_change(self):
        cfg, state = self.state()
        cfg["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _check_resume(state, cfg, {"manifest_sha256": "same"}, state["provenance"], Path("unused"))

    def test_resume_rejects_source_change(self):
        cfg, state = self.state()
        with self.assertRaisesRegex(ValueError, "Source"):
            _check_resume(state, cfg, {"manifest_sha256": "same"}, {"source_hashes": {}}, Path("unused"))

    def test_resume_rejects_manifest_change(self):
        cfg, state = self.state()
        with self.assertRaisesRegex(ValueError, "Manifest"):
            _check_resume(state, cfg, {"manifest_sha256": "changed"}, state["provenance"], Path("unused"))

    def test_cosine_resume_cannot_change_total_epochs(self):
        cfg, state = self.state()
        cfg["scheduler"] = state["config"]["scheduler"] = "cosine"
        cfg["max_epochs"] += 1
        with self.assertRaisesRegex(ValueError, "max_epochs"):
            _check_resume(state, cfg, {"manifest_sha256": "same"}, state["provenance"], Path("unused"))

    def test_lr_has_warmup_and_final_minimum(self):
        cfg = resolve_extended_config({**DEFAULT_CONFIG, "scheduler": "cosine", "warmup_epochs": 5})
        self.assertAlmostEqual(learning_rate_at(1, cfg), cfg["learning_rate"] / 5)
        self.assertAlmostEqual(learning_rate_at(cfg["max_epochs"], cfg), 1e-6)

    def test_statistics_align_image_ids(self):
        first = pd.DataFrame({"image_id": ["a", "b", "c"], "dice": [.2, .4, .6]})
        second = first.iloc[::-1].copy()
        result = paired_statistics(first, second)
        self.assertEqual(result["delta"], 0.)
        self.assertEqual(result["images_equal"], 3)
        self.assertEqual(result["p_wilcoxon"], 1.)

    def test_custom_run_name_appears_in_demo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "C_v2"
            run.mkdir(parents=True)
            (run / "best.pt").touch()
            (run / "summary.json").write_text(json.dumps(dict(
                trainer_kind="extension_20261111", status="completed", model_name="unetplusplus",
                config={"model_kwargs": {"deep_supervision": True}})), encoding="utf-8")
            result = append_extended_runs([], root)
            self.assertEqual(len(result), 1)
            self.assertIn("C_v2", result[0]["display_name"])


if __name__ == "__main__":
    unittest.main()
