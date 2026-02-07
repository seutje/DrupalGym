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


if __name__ == "__main__":
    unittest.main()
