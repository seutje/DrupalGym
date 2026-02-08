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

    def test_ambiguous_instruction_input_pair_rejected(self):
        gate = QualityGate(
            _DummyLogger(),
            config={
                "run_php_lint": False,
                "min_output_chars": 10,
                "max_output_chars": 5000,
                "reject_path_leakage_tokens": False,
                "reject_ambiguous_instruction_input": True,
                "max_outputs_per_instruction_input": 1,
            },
        )
        first = {
            "instruction": "Provide the Drupal 11 YAML configuration from <source_file>.",
            "input": "Source file: <source_file>\nFile name hint: one.yml\n",
            "output": "name: one\nvalue: 1\nstatus: true\n",
            "metadata": {"source": "docs/www_drupal_org/one.yml", "type": "yaml_reference"},
        }
        second = {
            "instruction": "Provide the Drupal 11 YAML configuration from <source_file>.",
            "input": "Source file: <source_file>\nFile name hint: one.yml\n",
            "output": "name: two\nvalue: 2\nstatus: true\n",
            "metadata": {"source": "docs/www_drupal_org/two.yml", "type": "yaml_reference"},
        }
        ok1, reason1 = gate.check_sample(first)
        ok2, reason2 = gate.check_sample(second)
        self.assertTrue(ok1)
        self.assertEqual(reason1, "")
        self.assertFalse(ok2)
        self.assertEqual(reason2, "ambiguous_instruction_input_pair")

    def test_missing_context_input_rejected_for_required_type(self):
        gate = QualityGate(
            _DummyLogger(),
            config={
                "run_php_lint": False,
                "min_output_chars": 10,
                "max_output_chars": 5000,
                "require_non_empty_input_for_types": ["yaml_reference"],
            },
        )
        sample = {
            "instruction": "Provide the Drupal 11 YAML configuration from <source_file>.",
            "input": "",
            "output": "name: DrupalGym\nstatus: true\n",
            "metadata": {"source": "docs/www_drupal_org/site.yml", "type": "yaml_reference"},
        }
        ok, reason = gate.check_sample(sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_context_input")

    def test_doc_source_allowlist_and_topic_denylist(self):
        gate = QualityGate(
            _DummyLogger(),
            config={
                "run_php_lint": False,
                "min_output_chars": 10,
                "max_output_chars": 5000,
                "doc_source_allowlist_prefixes": ["docs/www_drupal_org/"],
                "doc_topic_denylist_terms": ["mcp"],
            },
        )
        bad_topic = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: MCP Client",
            "input": "Source file: <source_file>",
            "output": (
                "This content is long enough to pass length checks for this unit test case. "
                "It includes enough alphabetic characters and multiple explanatory phrases "
                "to avoid summary mismatch rejections."
            ),
            "metadata": {"source": "docs/www_drupal_org/docs/mcp.md", "type": "doc_summary", "topic": "MCP Client"},
        }
        ok, reason = gate.check_sample(bad_topic)
        self.assertFalse(ok)
        self.assertEqual(reason, "doc_topic_denied")

        bad_source = {
            "instruction": "Explain the following topic based on Drupal 11 documentation: State API",
            "input": "Source file: <source_file>",
            "output": (
                "This content is long enough to pass length checks for this unit test case. "
                "It includes enough alphabetic characters and multiple explanatory phrases "
                "to avoid summary mismatch rejections."
            ),
            "metadata": {"source": "repos/mcp_tools/README.md", "type": "doc_summary", "topic": "State API"},
        }
        ok, reason = gate.check_sample(bad_source)
        self.assertFalse(ok)
        self.assertEqual(reason, "doc_source_not_allowed")


if __name__ == "__main__":
    unittest.main()
