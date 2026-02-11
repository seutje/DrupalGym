import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.train import (
    _audit_dataset_artifacts,
    _build_model_config_with_compat_fixes,
    _build_completion_data_collator,
    _build_completion_labels,
    _completion_marker_for_prompt_template,
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
        labels = _build_completion_labels(token_ids, marker_tokens)
        self.assertEqual(labels[:4], [-100, -100, -100, -100])
        self.assertEqual(labels[4:], [50, 60])

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

    def test_raise_actionable_model_load_error_reraises_other_errors(self):
        original = RuntimeError("boom")
        with self.assertRaises(RuntimeError) as ctx:
            _raise_actionable_model_load_error(original, "any/model")
        self.assertIs(ctx.exception, original)

    def test_build_model_config_with_compat_fixes_uses_regular_autoconfig(self):
        sentinel = object()

        class _AutoConfigOK:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                return sentinel

        cfg = _build_model_config_with_compat_fixes(
            model_name="any/model",
            logger=_DummyLogger(),
            auto_config_cls=_AutoConfigOK,
        )
        self.assertIs(cfg, sentinel)

    def test_build_model_config_with_compat_fixes_patches_ministral3_text_config(self):
        logger = _CapturingLogger()

        class _AutoConfigBroken:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                raise KeyError("ministral3")

            @staticmethod
            def get_config_dict(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                return (
                    {
                        "model_type": "mistral3",
                        "text_config": {"model_type": "ministral3", "hidden_size": 128},
                    },
                    {},
                )

        class _Mistral3ConfigStub:
            @staticmethod
            def from_dict(config_dict):
                return config_dict

        cfg = _build_model_config_with_compat_fixes(
            model_name="mistralai/Ministral-3-3B-Instruct-2512",
            logger=logger,
            auto_config_cls=_AutoConfigBroken,
            mistral3_config_cls=_Mistral3ConfigStub,
        )
        self.assertEqual(cfg["text_config"]["model_type"], "mistral")
        self.assertTrue(any("compatibility patch" in event[1] for event in logger.events))

    def test_build_model_config_with_compat_fixes_loader_fallback_without_get_config_dict(self):
        logger = _CapturingLogger()

        class _AutoConfigNoGetter:
            @staticmethod
            def from_pretrained(_model_name, trust_remote_code=True):
                self.assertTrue(trust_remote_code)
                raise KeyError("ministral3")

        class _Mistral3ConfigStub:
            @staticmethod
            def from_dict(config_dict):
                return config_dict

        def _loader(_model_name):
            return {
                "model_type": "mistral3",
                "text_config": {"model_type": "ministral3", "hidden_size": 256},
            }

        cfg = _build_model_config_with_compat_fixes(
            model_name="mistralai/Ministral-3-3B-Instruct-2512",
            logger=logger,
            auto_config_cls=_AutoConfigNoGetter,
            mistral3_config_cls=_Mistral3ConfigStub,
            config_dict_loader=_loader,
        )
        self.assertEqual(cfg["text_config"]["model_type"], "mistral")


if __name__ == "__main__":
    unittest.main()
