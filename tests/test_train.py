import json
import tempfile
import unittest
from pathlib import Path

from pipeline.train import _audit_dataset_artifacts, _build_completion_labels


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


if __name__ == "__main__":
    unittest.main()
