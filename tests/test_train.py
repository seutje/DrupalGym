import json
import tempfile
import unittest
from pathlib import Path

from pipeline.train import (
    _audit_dataset_artifacts,
    _build_completion_data_collator,
    _build_completion_labels,
)


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


class TrainHelpersTest(unittest.TestCase):
    def test_completion_labels_mask_prompt_tokens(self):
        marker_tokens = [30, 40]
        token_ids = [10, 20, 30, 40, 50, 60]
        labels = _build_completion_labels(token_ids, marker_tokens)
        self.assertEqual(labels[:4], [-100, -100, -100, -100])
        self.assertEqual(labels[4:], [50, 60])

    def test_dataset_artifact_audit_detects_numeric_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            valid_path = root / "valid.jsonl"
            bad_sample = {
                "instruction": "Explain the following topic based on Drupal 11 documentation: Example",
                "input": "",
                "output": "\n".join(str(i) for i in range(1, 20)),
            }
            good_sample = {
                "instruction": "Explain the following topic based on Drupal 11 documentation: Routing",
                "input": "",
                "output": "Use routing.yml with controller class and proper permissions.",
            }
            with open(train_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(bad_sample) + "\n")
            with open(valid_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(good_sample) + "\n")

            ok = _audit_dataset_artifacts(
                dataset_dir=root,
                logger=_DummyLogger(),
                max_numeric_line_streak=12,
                max_repeated_line_ratio=0.15,
            )
            self.assertFalse(ok)

    def test_completion_data_collator_pads_mixed_lengths(self):
        collator = _build_completion_data_collator(
            pad_token_id=42,
            padding_strategy="dynamic",
            pad_to_multiple_of=None,
        )
        features = [
            {
                "input_ids": [1] * 2048,
                "attention_mask": [1] * 2048,
                "labels": [7] * 2048,
            },
            {
                "input_ids": [2] * 2047,
                "attention_mask": [1] * 2047,
                "labels": [8] * 2047,
            },
        ]

        batch = collator(features)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2048))
        self.assertEqual(tuple(batch["attention_mask"].shape), (2, 2048))
        self.assertEqual(tuple(batch["labels"].shape), (2, 2048))
        self.assertEqual(batch["input_ids"][1, -1].item(), 42)
        self.assertEqual(batch["attention_mask"][1, -1].item(), 0)
        self.assertEqual(batch["labels"][1, -1].item(), -100)

    def test_completion_data_collator_pad_to_multiple_of(self):
        collator = _build_completion_data_collator(
            pad_token_id=0,
            padding_strategy="dynamic",
            pad_to_multiple_of=8,
        )
        features = [
            {
                "input_ids": [1] * 15,
                "attention_mask": [1] * 15,
                "labels": [2] * 15,
            },
            {
                "input_ids": [3] * 13,
                "attention_mask": [1] * 13,
                "labels": [4] * 13,
            },
        ]

        batch = collator(features)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 16))
        self.assertEqual(batch["labels"][0, -1].item(), -100)


if __name__ == "__main__":
    unittest.main()
