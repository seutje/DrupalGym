import unittest

from pipeline.dataset_refinement import (
    _chunk_sample,
    _rebalance_test_ratio,
    _validate_sample,
)


class DatasetRefinementHelpersTest(unittest.TestCase):
    def test_validate_rejects_malformed_class_slot(self):
        sample = {
            "instruction": "Show me the implementation of the class is in the file repos/drupal_core/core/lib/Drupal/Foo.php.",
            "input": "",
            "output": "<?php\n\nnamespace Drupal\\Foo;\n\nclass Foo {}\n",
            "metadata": {"source": "repos/drupal_core/core/lib/Drupal/Foo.php"},
        }
        passed, reason = _validate_sample(sample)
        self.assertFalse(passed)
        self.assertEqual(reason, "malformed_instruction_class_slot")

    def test_validate_rejects_class_interface_mismatch(self):
        sample = {
            "instruction": "Show me the implementation of the class StreamWrapperManagerInterface in the file repos/drupal_core/core/lib/Drupal/Core/StreamWrapper/StreamWrapperManagerInterface.php.",
            "input": "",
            "output": "<?php\n\nnamespace Drupal\\Core\\StreamWrapper;\n\ninterface StreamWrapperManagerInterface {}\n",
            "metadata": {
                "source": "repos/drupal_core/core/lib/Drupal/Core/StreamWrapper/StreamWrapperManagerInterface.php"
            },
        }
        passed, reason = _validate_sample(sample)
        self.assertFalse(passed)
        self.assertEqual(reason, "class_interface_mismatch")

    def test_chunk_sample_splits_long_output(self):
        output = "\n".join([f"line_{idx}" for idx in range(1, 610)])
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
            "input": "",
            "output": output,
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        chunks = _chunk_sample(sample, max_output_lines=300, overlap_lines=30)
        self.assertGreater(len(chunks), 1)
        self.assertIn("[Part 1/", chunks[0]["instruction"])
        self.assertLessEqual(len(chunks[0]["output"].splitlines()), 300)

    def test_rebalance_reduces_test_ratio(self):
        samples = []
        for idx in range(10):
            samples.append(
                {
                    "instruction": f"prod_{idx}",
                    "input": "",
                    "output": "<?php\nclass Prod {}\n",
                    "metadata": {"source": f"repos/example/src/Prod{idx}.php"},
                }
            )
        for idx in range(20):
            samples.append(
                {
                    "instruction": f"test_{idx}",
                    "input": "",
                    "output": "<?php\nclass TestThing {}\n",
                    "metadata": {"source": f"repos/drupal_core/core/tests/src/Unit/Test{idx}.php"},
                }
            )

        rebalanced, dropped, before_ratio, after_ratio = _rebalance_test_ratio(
            samples, seed=42, target_test_ratio=0.3
        )
        self.assertEqual(len(rebalanced) + len(dropped), len(samples))
        self.assertGreater(before_ratio, after_ratio)
        self.assertLessEqual(after_ratio, 0.3)


if __name__ == "__main__":
    unittest.main()
