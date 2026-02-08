import unittest

from pipeline.quality_gates import QualityGate


class _DummyLogger:
    def error(self, *_args, **_kwargs):
        return None


class QualityGateHelpersTest(unittest.TestCase):
    def setUp(self):
        self.gate = QualityGate(
            _DummyLogger(),
            config={
                "run_php_lint": False,
                "min_output_chars": 10,
                "max_output_chars": 5000,
                "min_output_chars_by_type": {"yaml_reference": 20},
                "max_output_chars_by_type": {"yaml_reference": 200},
            },
        )

    def test_invalid_symbol_prompt_rejected(self):
        sample = {
            "instruction": "Show me the implementation of the class broken in the file foo.php",
            "input": "",
            "output": "<?php\nnamespace Drupal\\Foo;\nclass Foo {}\n",
            "metadata": {"source": "foo.php"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "invalid_symbol_kind_prompt")

    def test_duplicate_output_rejected(self):
        first = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: Example",
            "input": "",
            "output": (
                "This is a sufficiently long sample output for the quality gate.\n"
                "It contains multiple lines to satisfy detail checks.\n"
                "It references Drupal 11 coding patterns and conventions."
            ),
            "metadata": {"source": "doc.md", "type": "doc_summary"},
        }
        second = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: Example 2",
            "input": "",
            "output": (
                "This is a sufficiently long sample output for the quality gate.\n"
                "It contains multiple lines to satisfy detail checks.\n"
                "It references Drupal 11 coding patterns and conventions."
            ),
            "metadata": {"source": "doc2.md", "type": "doc_summary"},
        }
        ok1, reason1 = self.gate.check_sample(first)
        ok2, reason2 = self.gate.check_sample(second)
        self.assertTrue(ok1)
        self.assertEqual(reason1, "")
        self.assertFalse(ok2)
        self.assertEqual(reason2, "near_duplicate_content")

    def test_prompt_wrapper_echo_rejected(self):
        sample = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: Services",
            "input": "",
            "output": (
                "Instruction: build a service.\n"
                "Input: logger.factory\n"
                "Output: use constructor injection and return PHP code.\n"
            ),
            "metadata": {"source": "doc.md"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "prompt_wrapper_echo")

    def test_numeric_line_streak_rejected(self):
        numbered = "\n".join(str(i) for i in range(1, 45))
        sample = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: Routing",
            "input": "",
            "output": numbered,
            "metadata": {"source": "doc.md"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "numeric_line_streak")

    def test_numeric_fenced_block_rejected(self):
        sample = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: Components",
            "input": "",
            "output": (
                "Drupal 11 component docs.\n"
                "```text\n"
                "1\n2\n3\n4\n5\n6\n7\n8\n"
                "```\n"
                "Use proper SDC patterns.\n"
            ),
            "metadata": {"source": "doc.md"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "numeric_code_block_artifact")

    def test_yaml_instruction_output_mismatch_rejected(self):
        sample = {
            "instruction": "Explain this file.",
            "input": "",
            "output": "name: DrupalGym\n",
            "metadata": {"source": "repos/example/foo.services.yml", "type": "yaml_reference"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "yaml_instruction_output_mismatch")

    def test_yaml_uses_type_specific_min_length(self):
        sample = {
            "instruction": "Provide the Drupal 11 YAML configuration from <source_file>.",
            "input": "",
            "output": "a: b\n",
            "metadata": {"source": "repos/example/foo.services.yml", "type": "yaml_reference"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "too_short")

    def test_procedural_php_without_namespace_allowed(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file <source_file>.",
            "input": "",
            "output": (
                "<?php\n"
                "function example_help(): string {\n"
                "  return 'Drupal 11 procedural module file output that is long enough.';\n"
                "}\n"
            ),
            "metadata": {"source": "repos/example/example.module", "type": "code_reference"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_path_leakage_in_model_fields_rejected(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/example.module.",
            "input": "",
            "output": (
                "<?php\n"
                "function example_help(): string {\n"
                "  return 'Drupal 11 procedural module file output that is long enough.';\n"
                "}\n"
            ),
            "metadata": {"source": "repos/example/example.module", "type": "code_reference"},
        }
        ok, reason = self.gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "path_leakage_token")


if __name__ == "__main__":
    unittest.main()
