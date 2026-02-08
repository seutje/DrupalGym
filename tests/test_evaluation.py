import unittest
from pathlib import Path

from pipeline.evaluation import (
    _build_generation_kwargs,
    _compute_format_sanity,
    _extract_code_blocks,
    _load_model_for_evaluation,
    _required_checks_for_prompt,
    _score_result,
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

    def test_required_checks_block_attribute(self):
        output = (
            "<?php\n"
            "#[Block(id: 'gym_stats', admin_label: 'Gym Statistics')]\n"
            "class GymStatsBlock {}\n"
        )
        checks, required = _required_checks_for_prompt("block_attribute", output)
        self.assertTrue(all(checks[name] for name in required))

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
