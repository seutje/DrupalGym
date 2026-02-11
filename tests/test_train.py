import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.train import (
    _audit_dataset_artifacts,
    _build_completion_marker_variants,
    _build_completion_data_collator,
    _build_completion_labels,
    _completion_marker_for_prompt_template,
    _ensure_native_ministral3_support,
    _normalize_ministral3_config_for_causallm,
    _coerce_model_config_object,
    _strip_pretrained_quantization_config,
    _format_training_text,
    _load_tokenizer_for_model,
    _missing_quality_tools,
    _raise_actionable_model_load_error,
    _resolve_prompt_template,
    PROMPT_TEMPLATE_MINISTRAL_INST,
    PROMPT_TEMPLATE_PLAIN,
)


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _DummyTokenizer:
    def __init__(self, pad_token=None, eos_token="</s>"):
        self.pad_token = pad_token
        self.eos_token = eos_token

    def __call__(self, text, add_special_tokens=False):
        if text == "[/INST]":
            return {"input_ids": [9]}
        if text == " [/INST]":
            return {"input_ids": [3, 9]}
        if text == "\n[/INST]":
            return {"input_ids": [4, 9]}
        return {"input_ids": [1, 2]}


class _CapturingLogger:
    def __init__(self):
        self.events = []

    def info(self, message, **kwargs):
        self.events.append(("INFO", message, kwargs))

    def error(self, message, **kwargs):
        self.events.append(("ERROR", message, kwargs))


class TrainHelpersTest(unittest.TestCase):
    def test_completion_labels_mask_prompt_tokens(self):
        marker_tokens = [30, 40]
        token_ids = [10, 20, 30, 40, 50, 60]
        labels, found = _build_completion_labels(token_ids, [marker_tokens])
        self.assertTrue(found)
        self.assertEqual(labels[:4], [-100, -100, -100, -100])
        self.assertEqual(labels[4:], [50, 60])

    def test_completion_labels_fallback_when_marker_missing(self):
        token_ids = [10, 20, 30, 40]
        labels, found = _build_completion_labels(token_ids, [[99]])
        self.assertFalse(found)
        self.assertEqual(labels, token_ids)

    def test_completion_marker_variants_include_whitespace_forms(self):
        variants = _build_completion_marker_variants(_DummyTokenizer(), "[/INST]")
        self.assertIn([9], variants)
        self.assertIn([3, 9], variants)
        self.assertIn([4, 9], variants)

    def test_prompt_template_autodetects_ministral_3(self):
        model_cfg = {
            "name": "Ministral-3-3B-Test",
            "base_model": "mistralai/Ministral-3-3B-Instruct-2512",
        }
        self.assertEqual(_resolve_prompt_template(model_cfg), PROMPT_TEMPLATE_MINISTRAL_INST)

    def test_prompt_template_defaults_to_plain(self):
        model_cfg = {
            "name": "Qwen2.5-Coder-7B",
            "base_model": "Qwen/Qwen2.5-Coder-7B",
        }
        self.assertEqual(_resolve_prompt_template(model_cfg), PROMPT_TEMPLATE_PLAIN)

    def test_ministral_inst_format_and_marker(self):
        text = _format_training_text(
            "Do X",
            "Given Y",
            "Here is Z",
            prompt_template=PROMPT_TEMPLATE_MINISTRAL_INST,
        )
        self.assertEqual(text, "<s>[INST] Do X\n\nGiven Y [/INST] Here is Z</s>")
        self.assertEqual(
            _completion_marker_for_prompt_template(PROMPT_TEMPLATE_MINISTRAL_INST),
            "[/INST]",
        )

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

    def test_dataset_artifact_audit_detects_special_token_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            valid_path = root / "valid.jsonl"
            bad_sample = {
                "instruction": "Explain the following topic based on Drupal 11 documentation: Example",
                "input": "",
                "output": "Looks normal\n<|fim_suffix|>\nbut should fail\n",
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

    def test_dataset_artifact_audit_detects_plain_fim_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            valid_path = root / "valid.jsonl"
            bad_sample = {
                "instruction": "Explain the following topic based on Drupal 11 documentation: Example",
                "input": "",
                "output": "Looks normal\n<fim_middle>\nbut should fail\n",
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

    def test_dataset_artifact_audit_detects_wrapper_echo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_path = root / "train.jsonl"
            valid_path = root / "valid.jsonl"
            bad_sample = {
                "instruction": "Explain the following topic based on Drupal 11 documentation: Example",
                "input": "",
                "output": "### Instruction: bad\n### Response:\nexample\n",
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

    def test_missing_quality_tools_uses_composer_home_bin_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = Path(tmpdir) / "vendor" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for tool in ("phpcs", "phpstan"):
                tool_path = bin_dir / tool
                tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                tool_path.chmod(0o755)

            config = {
                "quality": {"run_phpcs": True, "run_phpstan": True},
                "evaluation": {"run_phpcs": False, "run_phpstan": False},
            }
            with mock.patch.dict(os.environ, {"COMPOSER_HOME": tmpdir}, clear=False):
                with mock.patch("pipeline.train.shutil.which", return_value=None):
                    missing = _missing_quality_tools(config)

        self.assertEqual(missing, [])

    def test_load_tokenizer_for_model_uses_auto_tokenizer(self):
        class _AutoTokenizerOK:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                return _DummyTokenizer(pad_token="<pad>")

        tokenizer = _load_tokenizer_for_model(
            model_name="mistralai/Ministral-3-3B-Instruct-2512",
            logger=_DummyLogger(),
            auto_tokenizer_cls=_AutoTokenizerOK,
        )
        self.assertEqual(tokenizer.pad_token, "<pad>")

    def test_load_tokenizer_for_model_uses_mistral_fallback(self):
        class _AutoTokenizerBroken:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                raise ValueError(
                    "Tokenizer class TokenizersBackend does not exist or is not currently imported."
                )

        class _MistralTokenizerOK:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                return _DummyTokenizer(pad_token=None, eos_token="</s>")

        tokenizer = _load_tokenizer_for_model(
            model_name="mistralai/Ministral-3-3B-Instruct-2512",
            logger=_DummyLogger(),
            auto_tokenizer_cls=_AutoTokenizerBroken,
            mistral_tokenizer_cls=_MistralTokenizerOK,
        )
        self.assertEqual(tokenizer.pad_token, "</s>")

    def test_load_tokenizer_for_model_preserves_non_matching_error(self):
        class _AutoTokenizerOtherError:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                raise ValueError("Some other tokenizer error")

        with self.assertRaises(ValueError):
            _load_tokenizer_for_model(
                model_name="any/model",
                logger=_DummyLogger(),
                auto_tokenizer_cls=_AutoTokenizerOtherError,
            )

    def test_raise_actionable_model_load_error_for_ministral3_keyerror(self):
        with self.assertRaises(RuntimeError) as ctx:
            _raise_actionable_model_load_error(KeyError("ministral3"), "mistralai/Ministral-3-3B-Instruct-2512")
        self.assertIn("model_type `ministral3`", str(ctx.exception))
        self.assertIn("transformers", str(ctx.exception))

    def test_raise_actionable_model_load_error_for_mistral3_message(self):
        err = ValueError("Unrecognized configuration class Mistral3Config")
        with self.assertRaises(RuntimeError) as ctx:
            _raise_actionable_model_load_error(err, "mistralai/Ministral-3-3B-Instruct-2512")
        self.assertIn("transformers>=5.0.0", str(ctx.exception))

    def test_raise_actionable_model_load_error_reraises_other_errors(self):
        original = RuntimeError("boom")
        with self.assertRaises(RuntimeError) as ctx:
            _raise_actionable_model_load_error(original, "any/model")
        self.assertIs(ctx.exception, original)

    def test_ensure_native_ministral3_support_non_ministral(self):
        _ensure_native_ministral3_support("Qwen/Qwen2.5-Coder-3B")

    def test_ensure_native_ministral3_support_rejects_old_transformers(self):
        with mock.patch("transformers.__version__", "4.57.6"):
            with self.assertRaises(RuntimeError) as ctx:
                _ensure_native_ministral3_support("mistralai/Ministral-3-3B-Instruct-2512")
        self.assertIn(">=5.0.0", str(ctx.exception))

    def test_ensure_native_ministral3_support_accepts_new_transformers(self):
        with mock.patch("transformers.__version__", "5.1.0"):
            _ensure_native_ministral3_support("mistralai/Ministral-3-3B-Instruct-2512")

    def test_ensure_native_ministral3_support_rejects_transformers_v5(self):
        with mock.patch("transformers.__version__", "4.99.0"):
            with self.assertRaises(RuntimeError) as ctx:
                _ensure_native_ministral3_support("mistralai/Ministral-3-3B-Instruct-2512")
        self.assertIn(">=5.0.0", str(ctx.exception))

    def test_normalize_ministral3_config_for_causallm_converts_mistral3(self):
        class _Mistral3ConfigStub:
            def to_dict(self):
                return {"model_type": "mistral3", "hidden_size": 123}

        _Mistral3ConfigStub.__name__ = "Mistral3Config"

        class _Ministral3ConfigStub:
            @staticmethod
            def from_dict(data):
                return {"normalized": True, "data": data}

        with mock.patch("transformers.Ministral3Config", _Ministral3ConfigStub):
            out = _normalize_ministral3_config_for_causallm(
                "mistralai/Ministral-3-3B-Instruct-2512",
                _Mistral3ConfigStub(),
                _DummyLogger(),
            )
        self.assertTrue(out["normalized"])

    def test_strip_pretrained_quantization_config_rebuilds_config_without_key(self):
        class _ConfigStub:
            def __init__(self, payload):
                self.payload = payload

            def to_dict(self):
                return dict(self.payload)

            @classmethod
            def from_dict(cls, data):
                obj = cls(data)
                return obj

        cfg = _ConfigStub({"model_type": "x", "quantization_config": {"quant_method": "fp8"}})
        out = _strip_pretrained_quantization_config(cfg, _DummyLogger())
        self.assertNotIn("quantization_config", out.to_dict())

    def test_strip_pretrained_quantization_config_noop_without_key(self):
        class _ConfigStub:
            def to_dict(self):
                return {"model_type": "x"}

            @classmethod
            def from_dict(cls, data):
                return cls()

        cfg = _ConfigStub()
        out = _strip_pretrained_quantization_config(cfg, _DummyLogger())
        self.assertIs(out, cfg)

    def test_coerce_model_config_object_rebuilds_dict_ministral3(self):
        class _Ministral3ConfigStub:
            @staticmethod
            def from_dict(data):
                return {"rebuilt": True, "model_type": data.get("model_type")}

        with mock.patch("transformers.Ministral3Config", _Ministral3ConfigStub):
            out = _coerce_model_config_object({"model_type": "ministral3"}, _DummyLogger())
        self.assertTrue(out["rebuilt"])


if __name__ == "__main__":
    unittest.main()
