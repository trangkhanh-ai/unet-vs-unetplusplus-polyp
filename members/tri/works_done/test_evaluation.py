"""Kiểm tra giao thức và đường đi dự đoán bằng ảnh tổng hợp, không đọc test thật."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from members.duy.works_done.model_factory import build_model
from .checkpoint_io import load_checkpoint, sha256_file, verify_protocol_lock
from .evaluate import evaluate_checkpoint
from .predict import predict_image
from .compare import check_comparable, compare_evaluations


def synthetic_checkpoint(manifest):
    config = {"model":"unet", "model_kwargs":{"in_channels":3,"out_channels":1,"base_channels":2,"batch_norm":False,"deep_supervision":False},
              "manifest":str(manifest),"seed":42,"image_size":[32,32],"evaluation":{"threshold":.5,"head_policy":"last"}}
    model = build_model(config["model"],**config["model_kwargs"])
    return {"format_version":1,"model_name":"unet","model_kwargs":config["model_kwargs"],"model_state":model.state_dict(),
            "config":config,"preprocessing":{"image_size":[32,32],"normalization":"rgb_float_0_1","mask":"binary_0_1"},
            "manifest_sha256":sha256_file(manifest),"epoch":1,"split_counts":{"train":1,"val":1,"test":1},
            "status":"synthetic_test_fixture","best":{"epoch":1},"provenance":{"source_hashes":{}}}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # Dataset bảo vệ project_root: fixture cũng đặt trong repo, không nới
        # kiểm tra đường dẫn chỉ để test chạy được.
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temporary.name)
        image_path, mask_path = self.root/"ảnh.png", self.root/"mask.png"
        rgb = np.zeros((48,64,3),dtype=np.uint8)
        rgb[:24,:32,0] = 255
        truth = np.zeros((48,64),dtype=np.uint8)
        truth[:24,:32] = 255
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(truth).save(mask_path)
        self.manifest = self.root/"splits.csv"
        pd.DataFrame([{"image_id":f"synthetic_{split}","image_path":str(image_path),"mask_path":str(mask_path),
            "split":split,"height":48,"width":64,"polyp_category":"large","polyp_area_ratio":.25}
            for split in ("train","val","test")]).to_csv(self.manifest,index=False)
        self.checkpoint = synthetic_checkpoint(self.manifest)
        self.checkpoint_path = self.root/"best.pt"
        torch.save(self.checkpoint,self.checkpoint_path)

    def tearDown(self):
        if not self.root.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise RuntimeError("Refuse cleanup outside this test's owned directory")
        self.temporary.cleanup()

    def make_lock(self):
        lock = {"schema_version":1,"status":"locked_before_test","created_at":"synthetic fixture timestamp",
            "manifest_sha256":self.checkpoint["manifest_sha256"],"split_counts":self.checkpoint["split_counts"],
            "protocol":{"image_size":[32,32],"threshold":.5,"head_policy":"last","base_channels":2,"batch_norm":False},
            "experiments":[{"label":"A","model":"unet","seed":42,"checkpoint_path":str(self.checkpoint_path),
                "checkpoint_sha256":sha256_file(self.checkpoint_path),"config":self.checkpoint["config"]}]}
        path = self.root/"lock.json"
        path.write_text(json.dumps(lock),encoding="utf-8")
        return lock,path

    def test_safe_load_and_unicode_original_prediction_size(self):
        model,checkpoint = load_checkpoint(self.checkpoint_path)
        image = Image.fromarray(np.full((48,64,3),127,dtype=np.uint8))
        result = predict_image(model,checkpoint,image)
        self.assertEqual(result["mask_original"].size,(64,48))
        self.assertEqual(result["mask_input"].size,(32,32))
        self.assertTrue(np.isfinite(result["probability_original"]).all())

    def test_test_requires_lock_before_loader_created(self):
        with patch("members.tri.works_done.evaluate.get_dataloader") as mocked:
            with self.assertRaisesRegex(ValueError,"protocol-lock"):
                evaluate_checkpoint(self.checkpoint_path,"test",self.root/"output",device="cpu")
            mocked.assert_not_called()

    def test_lock_detects_config_and_checkpoint_tampering(self):
        lock,path = self.make_lock()
        result = verify_protocol_lock(self.checkpoint,path,self.checkpoint_path)
        self.assertEqual(result["label"],"A")
        altered = copy.deepcopy(self.checkpoint)
        altered["config"]["seed"] = 7
        with self.assertRaisesRegex(ValueError,"config"):
            verify_protocol_lock(altered,path,self.checkpoint_path)
        self.checkpoint_path.write_bytes(self.checkpoint_path.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError,"authorize"):
            verify_protocol_lock(self.checkpoint,path,self.checkpoint_path)

    def test_changed_manifest_rejected_before_loader(self):
        self.manifest.write_text(self.manifest.read_text(encoding="utf-8")+"\n",encoding="utf-8")
        with patch("members.tri.works_done.evaluate.get_dataloader") as mocked:
            with self.assertRaisesRegex(ValueError,"manifest changed"):
                evaluate_checkpoint(self.checkpoint_path,"val",self.root/"output",device="cpu")
            mocked.assert_not_called()

    def test_synthetic_validation_exports_all_images_masks_and_means(self):
        result = evaluate_checkpoint(self.checkpoint_path,"val",self.root/"output",device="cpu",batch_size=1,warmup_batches=0)
        self.assertEqual(result["image_count"],1)
        self.assertEqual(result["split"],"val")
        self.assertIsNone(result["protocol_authorization"])
        table = pd.read_csv(self.root/"output"/"per_image.csv")
        self.assertEqual(table.loc[0,"image_id"],"synthetic_val")
        self.assertAlmostEqual(table["dice"].mean(),result["metrics"]["dice"])
        self.assertTrue((self.root/"output"/"masks"/"synthetic_val.png").exists())
        self.assertTrue((self.root/"output"/"qualitative.png").exists())
        modified = copy.deepcopy(result)
        modified["threshold"] = .6
        with self.assertRaisesRegex(ValueError,"threshold"):
            check_comparable([result,modified],[table,table])

    def test_complete_synthetic_comparison_exports_paired_images_and_csv(self):
        # Lặp cùng fixture để kiểm tra tích hợp xuất bảng/ảnh; đây không phải
        # bằng chứng U-Net++ tốt hơn U-Net và không giữ lại các kết quả fixture.
        folders = [self.root/"fixture_A",self.root/"fixture_B"]
        for folder in folders:
            evaluate_checkpoint(self.checkpoint_path,"val",folder,device="cpu",batch_size=1,warmup_batches=0)
        report = compare_evaluations(folders,self.root/"comparison")
        self.assertEqual(report["image_count"],1)
        self.assertEqual(report["paired_differences"]["fixture_B"]["mean_delta_dice"],0)
        self.assertTrue((self.root/"comparison"/"comparison.csv").is_file())
        self.assertTrue((self.root/"comparison"/"paired_examples.png").is_file())
        self.assertTrue((self.root/"comparison"/"comparison_by_size.csv").is_file())


if __name__ == "__main__":
    unittest.main()
