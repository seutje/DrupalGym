import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.evaluation import (
    _apply_external_required_checks,
    _artifact_guard,
    _build_generation_kwargs,
    _compute_format_sanity,
    _extract_code_blocks,
    _load_model_for_evaluation,
    _reset_sample_outputs_dir,
    _required_checks_for_prompt,
    _run_external_checks,
    _run_phpcs,
    _score_result,
    _select_snippets_for_checks,
    summarize_results,
)


class EvaluationHelpersTest(unittest.TestCase):
    def test_build_generation_kwargs_includes_repetition_controls(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        kwargs = _build_generation_kwargs(
            FakeTokenizer(),
            max_new_tokens=256,
            eval_cfg={"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
        )
        self.assertEqual(kwargs["max_new_tokens"], 256)
        self.assertEqual(kwargs["repetition_penalty"], 1.1)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 4)

    def test_extract_code_blocks_from_fences(self):
        text = (
            "Example\n"
            "```php\n<?php\nclass A {}\n```\n"
            "More text\n"
            "```yaml\nservices:\n  gym.service: {}\n```\n"
        )
        blocks = _extract_code_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("class A", blocks[0])
        self.assertIn("services:", blocks[1])

    def test_select_snippets_php_only_ignores_non_php_fences(self):
        text = (
            "```yaml\nservices:\n  gym.service: {}\n```\n"
            "```twig\n<div>{{ content }}</div>\n```\n"
            "```php\n<?php\nfinal class GymBlock {}\n```\n"
        )
        snippets, metadata = _select_snippets_for_checks(
            text,
            {"php_snippet_policy": "php_only", "max_code_checks_per_response": 5},
        )
        self.assertEqual(metadata["php_selection_policy"], "php_only")
        self.assertEqual(metadata["code_block_count"], 3)
        self.assertEqual(metadata["php_candidate_count"], 1)
        self.assertEqual(metadata["php_checked_count"], 1)
        self.assertEqual(len(snippets), 1)
        self.assertIn("final class GymBlock", snippets[0])

    def test_select_snippets_php_only_accepts_unlabeled_fence_with_php_tag(self):
        text = "```\n<?php\necho 'hello';\n```\n"
        snippets, metadata = _select_snippets_for_checks(
            text,
            {"php_snippet_policy": "php_only", "max_code_checks_per_response": 5},
        )
        self.assertEqual(metadata["php_candidate_count"], 1)
        self.assertEqual(metadata["php_checked_count"], 1)
        self.assertEqual(len(snippets), 1)
        self.assertTrue(snippets[0].lstrip().startswith("<?php"))

    def test_select_snippets_all_fences_keeps_legacy_behavior(self):
        text = (
            "```yaml\nservices:\n  gym.service: {}\n```\n"
            "```php\n<?php\necho 'ok';\n```\n"
        )
        snippets, metadata = _select_snippets_for_checks(
            text,
            {"php_snippet_policy": "all_fences", "max_code_checks_per_response": 5},
        )
        self.assertEqual(metadata["php_selection_policy"], "all_fences")
        self.assertEqual(metadata["code_block_count"], 2)
        self.assertEqual(metadata["php_candidate_count"], 2)
        self.assertEqual(metadata["php_checked_count"], 2)
        self.assertEqual(len(snippets), 2)

    def test_select_snippets_php_only_falls_back_to_inline_php_without_fences(self):
        text = "<?php\nfinal class InlineOnly {}\n"
        snippets, metadata = _select_snippets_for_checks(
            text,
            {"php_snippet_policy": "php_only", "max_code_checks_per_response": 5},
        )
        self.assertEqual(metadata["code_block_count"], 0)
        self.assertEqual(metadata["php_candidate_count"], 1)
        self.assertEqual(metadata["php_checked_count"], 1)
        self.assertEqual(len(snippets), 1)
        self.assertIn("InlineOnly", snippets[0])

    def test_run_external_checks_php_only_skips_when_no_php_snippets(self):
        output = (
            "```yaml\nservices:\n  gym.service: {}\n```\n"
            "```twig\n<div>{{ content }}</div>\n```\n"
        )

        def _lint_summary(snippets):
            return {
                "enabled": True,
                "available": True,
                "checked": len(snippets),
                "passed": len(snippets),
                "failed": 0,
                "errors": [],
            }

        def _phpcs_summary(snippets):
            return {
                "enabled": True,
                "available": True,
                "drupal_standard_available": True,
                "runtime_broken": False,
                "checked": len(snippets),
                "passed": len(snippets),
                "failed": 0,
                "errors": [],
            }

        def _phpstan_summary(snippets, failure_mode="syntax_only"):
            return {
                "enabled": True,
                "available": True,
                "failure_mode": failure_mode,
                "checked": len(snippets),
                "passed": len(snippets),
                "failed": 0,
                "syntax_errors": 0,
                "errors": [],
            }

        with (
            patch("pipeline.evaluation._run_php_lint", side_effect=_lint_summary),
            patch("pipeline.evaluation._run_phpcs", side_effect=_phpcs_summary),
            patch("pipeline.evaluation._run_phpstan", side_effect=_phpstan_summary),
        ):
            external = _run_external_checks(
                output,
                {
                    "run_php_lint": True,
                    "run_phpcs": True,
                    "run_phpstan": True,
                    "phpstan_failure_mode": "syntax_only",
                    "php_snippet_policy": "php_only",
                    "max_code_checks_per_response": 5,
                },
            )

        self.assertEqual(external["code_block_count"], 2)
        self.assertEqual(external["php_candidate_count"], 0)
        self.assertEqual(external["php_checked_count"], 0)
        self.assertEqual(external["php_lint"]["checked"], 0)
        self.assertEqual(external["phpcs"]["checked"], 0)
        self.assertEqual(external["phpstan"]["checked"], 0)

    def test_reset_sample_outputs_dir_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_outputs_dir = Path(tmp_dir) / "sample_outputs"
            stale_dir = sample_outputs_dir / "old_model"
            stale_dir.mkdir(parents=True)
            (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

            _reset_sample_outputs_dir(sample_outputs_dir)

            self.assertTrue(sample_outputs_dir.exists())
            self.assertEqual(list(sample_outputs_dir.iterdir()), [])

    def test_required_checks_block_attribute(self):
        output = (
            "<?php\n"
            "#[Block(id: 'gym_stats', admin_label: 'Gym Statistics')]\n"
            "class GymStatsBlock {}\n"
        )
        checks, required = _required_checks_for_prompt("block_attribute", output)
        self.assertTrue(all(checks[name] for name in required))

    def test_required_checks_can_enforce_fenced_php_contract(self):
        prompt = {"require_fenced_php": True}
        checks, required = _required_checks_for_prompt("custom_php_prompt", "<?php\nfinal class InlineOnly {}\n", prompt=prompt)
        self.assertIn("has_fenced_php_block", required)
        self.assertFalse(checks["has_fenced_php_block"])

    def test_apply_external_required_checks_requires_php_snippet_for_service_di(self):
        checks = {"non_empty_output": True}
        required = ["non_empty_output"]
        updated_checks, updated_required = _apply_external_required_checks(
            prompt_id="service_di",
            prompt={"requires_php": True},
            checks=checks,
            required=required,
            external_checks={"php_checked_count": 0},
        )
        self.assertIn("has_php_snippet", updated_required)
        self.assertFalse(updated_checks["has_php_snippet"])

    def test_apply_external_required_checks_passes_with_php_snippet(self):
        checks = {"non_empty_output": True}
        required = ["non_empty_output"]
        updated_checks, updated_required = _apply_external_required_checks(
            prompt_id="routing_yaml",
            prompt={"requires_php": True},
            checks=checks,
            required=required,
            external_checks={"php_checked_count": 1},
        )
        self.assertIn("has_php_snippet", updated_required)
        self.assertTrue(updated_checks["has_php_snippet"])

    def test_score_result_penalizes_failed_php_lint(self):
        required_checks = {
            "non_empty_output": True,
            "has_php_tag": True,
        }
        required = ["non_empty_output", "has_php_tag"]
        external = {
            "php_lint": {
                "enabled": True,
                "available": True,
                "checked": 1,
                "passed": 0,
                "failed": 1,
                "errors": [{"snippet": 1, "message": "Parse error"}],
            },
            "phpcs": {
                "enabled": False,
                "available": False,
                "drupal_standard_available": False,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
        }

        score = _score_result(required_checks, required, external)
        self.assertTrue(score["passes_required"])
        self.assertFalse(score["passes_php_lint"])
        self.assertFalse(score["passed"])
        self.assertLess(score["score"], 1.0)

    def test_score_result_phpcs_failure_is_style_only(self):
        required_checks = {"non_empty_output": True}
        required = ["non_empty_output"]
        external = {
            "php_lint": {
                "enabled": False,
                "available": False,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpcs": {
                "enabled": True,
                "available": True,
                "drupal_standard_available": True,
                "checked": 1,
                "passed": 0,
                "failed": 1,
                "errors": [{"snippet": 1, "message": "line 1: style issue"}],
            },
            "phpstan": {
                "enabled": False,
                "available": False,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
        }
        score = _score_result(required_checks, required, external)
        self.assertTrue(score["passes_semantic"])
        self.assertFalse(score["passes_style"])
        self.assertTrue(score["passed"])
        self.assertLess(score["style_score"], 1.0)

    def test_score_result_artifact_guard_hard_rejects(self):
        required_checks = {"non_empty_output": True}
        required = ["non_empty_output"]
        external = {
            "php_lint": {"enabled": False, "available": False, "checked": 0, "passed": 0, "failed": 0, "errors": []},
            "phpcs": {
                "enabled": False,
                "available": False,
                "drupal_standard_available": False,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpstan": {"enabled": False, "available": False, "checked": 0, "passed": 0, "failed": 0, "errors": []},
        }
        guard = _artifact_guard("### Instruction: bad wrapper leakage\n")
        score = _score_result(required_checks, required, external, artifact_guard=guard)
        self.assertFalse(score["passes_artifact_guard"])
        self.assertFalse(score["passes_semantic"])
        self.assertFalse(score["passed"])
        self.assertEqual(score["semantic_score"], 0.0)

    def test_score_result_fails_lint_gates_when_required_php_snippet_missing(self):
        required_checks = {
            "non_empty_output": True,
            "has_php_snippet": False,
        }
        required = ["non_empty_output", "has_php_snippet"]
        external = {
            "php_checked_count": 0,
            "php_lint": {
                "enabled": True,
                "available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpcs": {
                "enabled": True,
                "available": True,
                "drupal_standard_available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpstan": {
                "enabled": True,
                "available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
        }

        score = _score_result(required_checks, required, external)
        self.assertFalse(score["passes_php_lint"])
        self.assertFalse(score["passes_phpcs"])
        self.assertFalse(score["passes_phpstan"])
        self.assertFalse(score["passed"])
        self.assertLess(score["score"], 0.5)

    def test_score_result_keeps_lint_neutral_when_php_not_required_and_no_snippets(self):
        required_checks = {
            "non_empty_output": True,
        }
        required = ["non_empty_output"]
        external = {
            "php_checked_count": 0,
            "php_lint": {
                "enabled": True,
                "available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpcs": {
                "enabled": True,
                "available": True,
                "drupal_standard_available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
            "phpstan": {
                "enabled": True,
                "available": True,
                "checked": 0,
                "passed": 0,
                "failed": 0,
                "errors": [],
            },
        }

        score = _score_result(required_checks, required, external)
        self.assertTrue(score["passes_php_lint"])
        self.assertTrue(score["passes_phpcs"])
        self.assertTrue(score["passes_phpstan"])
        self.assertTrue(score["passes_required"])
        self.assertTrue(score["passed"])
        self.assertEqual(score["score"], 1.0)

    def test_run_phpcs_ignores_temp_filename_classname_noise(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as handle:
            tmp_path = Path(handle.name)
        report = {
            "totals": {"errors": 1, "warnings": 0, "fixable": 0},
            "files": {
                str(tmp_path): {
                    "errors": 1,
                    "warnings": 0,
                    "messages": [
                        {
                            "message": "Class name doesn't match filename; expected \"class tmpfile\"",
                            "source": "PSR1.Classes.ClassDeclaration.InvalidClassName",
                            "type": "ERROR",
                            "line": 1,
                            "column": 1,
                            "fixable": False,
                        }
                    ],
                }
            },
        }
        proc = SimpleNamespace(returncode=1, stdout=json.dumps(report), stderr="")

        with (
            patch("pipeline.evaluation.shutil.which", return_value="/usr/bin/phpcs"),
            patch("pipeline.evaluation._has_drupal_phpcs_standard", return_value=True),
            patch("pipeline.evaluation._write_temp_php", return_value=tmp_path),
            patch("pipeline.evaluation.subprocess.run", return_value=proc),
        ):
            summary = _run_phpcs(["class Example {}"])

        self.assertEqual(summary["checked"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["errors"], [])

    def test_summarize_results_tracks_prompt_deltas(self):
        results = [
            {
                "model_name": "QwenTest",
                "base_model": "Qwen/base",
                "variant": "fine_tuned",
                "prompt_id": "block_attribute",
                "category": "attributes",
                "score": 0.9,
                "passed": True,
                "format_sanity": {"score": 1.0, "is_sane": True},
            },
            {
                "model_name": "QwenTest",
                "base_model": "Qwen/base",
                "variant": "baseline",
                "prompt_id": "block_attribute",
                "category": "attributes",
                "score": 0.5,
                "passed": False,
                "format_sanity": {"score": 1.0, "is_sane": True},
            },
            {
                "model_name": "QwenTest",
                "base_model": "Qwen/base",
                "variant": "fine_tuned",
                "prompt_id": "service_di",
                "category": "di",
                "score": 0.7,
                "passed": True,
                "format_sanity": {"score": 0.4, "is_sane": False},
            },
            {
                "model_name": "QwenTest",
                "base_model": "Qwen/base",
                "variant": "baseline",
                "prompt_id": "service_di",
                "category": "di",
                "score": 0.7,
                "passed": True,
                "format_sanity": {"score": 1.0, "is_sane": True},
            },
        ]

        summary = summarize_results(results)
        self.assertEqual(summary["model_count"], 1)
        model_summary = summary["models"][0]
        self.assertEqual(model_summary["fine_tuned_wins"], 1)
        self.assertEqual(model_summary["ties"], 1)
        self.assertGreater(model_summary["delta_avg_score"], 0)
        self.assertIn("delta_avg_semantic_score", model_summary)
        self.assertIn("delta_avg_style_score", model_summary)
        self.assertLess(model_summary["fine_tuned_format_sanity_avg"], model_summary["baseline_format_sanity_avg"])
        self.assertGreater(model_summary["fine_tuned_format_sanity_fail_rate"], 0.0)

    def test_compute_format_sanity_flags_wrapper_and_numeric_artifact(self):
        output = "Instruction: test\n" + "\n".join(str(i) for i in range(1, 45))
        sanity = _compute_format_sanity(output)
        self.assertTrue(sanity["has_prompt_wrapper_echo"])
        self.assertGreaterEqual(sanity["numeric_line_streak"], 40)
        self.assertFalse(sanity["is_sane"])

    def test_model_load_falls_back_to_cpu_after_auto_failure(self):
        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class FakeBaseModel:
            def __init__(self, kwargs):
                self.kwargs = kwargs

        class FakeAutoModel:
            calls = []

            @classmethod
            def from_pretrained(cls, *_args, **kwargs):
                cls.calls.append(kwargs)
                if kwargs.get("device_map") == "auto":
                    raise RuntimeError("auto placement failed")
                return FakeBaseModel(kwargs)

        class FakePeftModelWrapper:
            def eval(self):
                return self

        class FakePeft:
            @classmethod
            def from_pretrained(cls, _base_model, _adapter_path):
                return FakePeftModelWrapper()

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeTorch:
            cuda = FakeCuda()
            float16 = "float16"
            float32 = "float32"

        class FakeLogger:
            def info(self, *_args, **_kwargs):
                return None

        tokenizer, base_model, model = _load_model_for_evaluation(
            model_name="QwenTest",
            base_model_id="Qwen/base",
            adapter_path=Path("models/QwenTest/test_run/adapter"),
            eval_cfg={"device": "auto"},
            eval_dir=Path("."),
            logger=FakeLogger(),
            torch_module=FakeTorch,
            auto_tokenizer_cls=FakeTokenizer,
            auto_model_cls=FakeAutoModel,
            peft_model_cls=FakePeft,
        )

        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(base_model)
        self.assertIsNotNone(model)
        self.assertGreaterEqual(len(FakeAutoModel.calls), 3)
        self.assertIn("offload_folder", FakeAutoModel.calls[0])
        self.assertEqual(FakeAutoModel.calls[-1]["device_map"], "cpu")


if __name__ == "__main__":
    unittest.main()
