import unittest

from pipeline.dataset_refinement import (
    _build_augmented_sample,
    _char_chunk_sample,
    _chunk_sample,
    _compile_category_patterns,
    _deduplicate_normalized_outputs,
    _enforce_source_share_cap,
    _is_augmentation_candidate,
    _sample_matches_category,
    _rebalance_test_ratio,
    _source_matches_prefix,
    _split_dataset,
    _validate_sample,
)


class DatasetRefinementHelpersTest(unittest.TestCase):
    def test_validate_rejects_invalid_symbol_kind_prompt(self):
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

    def test_validate_accepts_interface_prompt(self):
        sample = {
            "instruction": "Show me the implementation of the interface StreamWrapperManagerInterface in the file repos/drupal_core/core/lib/Drupal/Core/StreamWrapper/StreamWrapperManagerInterface.php.",
            "input": "",
            "output": "<?php\n\nnamespace Drupal\\Core\\StreamWrapper;\n\ninterface StreamWrapperManagerInterface {}\n",
            "metadata": {
                "source": "repos/drupal_core/core/lib/Drupal/Core/StreamWrapper/StreamWrapperManagerInterface.php"
            },
        }
        passed, reason = _validate_sample(sample)
        self.assertTrue(passed)
        self.assertEqual(reason, "")

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
        self.assertEqual(
            chunks[0]["instruction"],
            "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
        )
        self.assertEqual(chunks[0]["metadata"]["refinement"]["chunk_index"], 1)
        self.assertEqual(chunks[0]["metadata"]["refinement"]["chunk_total"], len(chunks))
        self.assertLessEqual(len(chunks[0]["output"].splitlines()), 300)

    def test_chunk_sample_suffix_mode_adds_instruction_marker(self):
        output = "\n".join([f"line_{idx}" for idx in range(1, 610)])
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
            "input": "",
            "output": output,
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        chunks = _chunk_sample(sample, max_output_lines=300, overlap_lines=30, instruction_mode="suffix")
        self.assertIn("[Part 1/", chunks[0]["instruction"])

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

    def test_split_dataset_keeps_source_groups_together(self):
        samples = []
        for idx in range(10):
            samples.append(
                {
                    "instruction": f"group_a_{idx}",
                    "input": "",
                    "output": "<?php\nclass A {}\n",
                    "metadata": {"source": "repos/example/src/A.php"},
                }
            )
        for idx in range(10):
            samples.append(
                {
                    "instruction": f"group_b_{idx}",
                    "input": "",
                    "output": "<?php\nclass B {}\n",
                    "metadata": {"source": "repos/example/src/B.php"},
                }
            )
        for idx in range(10):
            samples.append(
                {
                    "instruction": f"group_c_{idx}",
                    "input": "",
                    "output": "<?php\nclass C {}\n",
                    "metadata": {"source": "repos/example/src/C.php"},
                }
            )

        splits = _split_dataset(samples, targets={"train": 0.8, "valid": 0.1, "test": 0.1}, seed=42)
        split_by_source = {}
        for split_name, split_samples in splits.items():
            for sample in split_samples:
                source = sample["metadata"]["source"]
                split_by_source.setdefault(source, set()).add(split_name)

        for source, assigned_splits in split_by_source.items():
            self.assertEqual(
                len(assigned_splits),
                1,
                f"source {source} appears in multiple splits: {assigned_splits}",
            )

    def test_augmentation_candidate_rejects_non_php_and_chunked(self):
        non_php_sample = {
            "instruction": "sample",
            "input": "",
            "output": "# heading\n",
            "metadata": {"source": "repos/example/README.md"},
        }
        ok, reason = _is_augmentation_candidate(non_php_sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "augmentation_non_php_source")

        chunked_sample = {
            "instruction": "sample",
            "input": "",
            "output": "<?php\nclass Example {}\n",
            "metadata": {
                "source": "repos/example/src/Example.php",
                "refinement": {"chunk_index": 1, "chunk_total": 3},
            },
        }
        ok, reason = _is_augmentation_candidate(chunked_sample)
        self.assertFalse(ok)
        self.assertEqual(reason, "augmentation_chunked_source")

    def test_validate_rejects_prompt_wrapper_echo_in_output(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
            "input": "",
            "output": "Instruction: x\nInput: y\nOutput: z\n",
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        passed, reason = _validate_sample(sample)
        self.assertFalse(passed)
        self.assertEqual(reason, "contains_prompt_wrapper_echo")

    def test_validate_rejects_numeric_line_streak_artifact(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
            "input": "",
            "output": "\n".join(str(i) for i in range(1, 45)),
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        passed, reason = _validate_sample(sample)
        self.assertFalse(passed)
        self.assertEqual(reason, "numeric_line_streak_artifact")

    def test_validate_rejects_numeric_fenced_block_artifact(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file repos/example/src/Example.php.",
            "input": "",
            "output": "<?php\n```\n1\n2\n3\n4\n5\n6\n7\n8\n```\nclass Example {}\n",
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        passed, reason = _validate_sample(sample)
        self.assertFalse(passed)
        self.assertEqual(reason, "numeric_code_block_artifact")

    def test_source_prefix_match(self):
        sample = {
            "instruction": "Explain",
            "input": "",
            "output": "text",
            "metadata": {"source": "docs/symfony_com/doc/7.0/security/csrf.md"},
        }
        self.assertTrue(_source_matches_prefix(sample, ["docs/symfony_com/"]))
        self.assertFalse(_source_matches_prefix(sample, ["docs/drupal_org/"]))

    def test_enforce_source_share_cap_drops_overrepresented_bucket(self):
        samples = []
        for idx in range(90):
            samples.append(
                {
                    "instruction": f"a_{idx}",
                    "input": "",
                    "output": "<?php\nclass A {}\n",
                    "metadata": {"source": f"repos/drupal_core/core/src/A{idx}.php"},
                }
            )
        for idx in range(10):
            samples.append(
                {
                    "instruction": f"b_{idx}",
                    "input": "",
                    "output": "<?php\nclass B {}\n",
                    "metadata": {"source": f"repos/example/src/B{idx}.php"},
                }
            )

        kept, dropped, concentration = _enforce_source_share_cap(samples, max_source_share=0.45, seed=42)
        self.assertGreater(len(dropped), 0)
        core = next(item for item in concentration if item["source_prefix"] == "repos/drupal_core")
        self.assertLess(core["share"], 0.9)
        self.assertLessEqual(core["share"], 0.5)
        self.assertEqual(len(kept) + len(dropped), len(samples))

    def test_source_share_cap_preserves_category_matched_samples(self):
        samples = []
        for idx in range(80):
            output = "<?php\nclass CorePlain {}\n"
            if idx < 5:
                output = "<?php\n#[Block]\nclass CoreSdcLike {}\n"
            samples.append(
                {
                    "instruction": f"core_{idx}",
                    "input": "",
                    "output": output,
                    "metadata": {"source": f"repos/drupal_core/core/src/Core{idx}.php", "sample_type": "retrieval"},
                }
            )
        for idx in range(20):
            samples.append(
                {
                    "instruction": f"ext_{idx}",
                    "input": "",
                    "output": "<?php\nclass Ext {}\n",
                    "metadata": {"source": f"repos/example/src/Ext{idx}.php", "sample_type": "retrieval"},
                }
            )

        patterns = _compile_category_patterns({"attributes": [r"#\[[A-Za-z_\\][A-Za-z0-9_\\]*"]})
        kept, _dropped, _concentration = _enforce_source_share_cap(
            samples,
            max_source_share=0.55,
            seed=42,
            preserve_categories=["attributes"],
            category_patterns=patterns,
        )
        preserved = [
            sample
            for sample in kept
            if sample["metadata"]["source"].startswith("repos/drupal_core/")
            and "#[" in sample["output"]
        ]
        self.assertGreaterEqual(len(preserved), 5)

    def test_sample_matches_category(self):
        sample = {
            "instruction": "Create a Drupal 11 service with constructor injection.",
            "input": "",
            "output": "services:\n  gym.logger:\n    class: Drupal\\gym\\Logger\n",
            "metadata": {"source": "repos/example/gym.services.yml"},
        }
        patterns = _compile_category_patterns()
        self.assertTrue(_sample_matches_category(sample, "di", category_patterns=patterns))
        self.assertFalse(_sample_matches_category(sample, "sdc", category_patterns=patterns))

    def test_augmented_sample_hides_repo_paths_from_input(self):
        sample = {
            "instruction": "Show me the implementation of the class Example in the file <source_file>.",
            "input": "",
            "output": "<?php\n\nnamespace Drupal\\example;\n\nclass Example {}\n",
            "metadata": {"source": "repos/example/src/Example.php"},
        }
        augmented = _build_augmented_sample(
            sample,
            augmentation_type="write_from_spec",
            excerpt_lines=50,
            max_output_lines=300,
        )
        self.assertNotIn("repos/", augmented["input"])
        self.assertIn("<source_file>", augmented["input"])
        self.assertEqual(augmented["metadata"]["sample_type"], "write_from_spec")

    def test_validate_context_requirement_for_types(self):
        sample = {
            "instruction": "Provide the Drupal 11 YAML configuration from <source_file>.",
            "input": "",
            "output": "name: DrupalGym\nstatus: true\n",
            "metadata": {"source": "repos/example/example.services.yml", "type": "yaml_reference"},
        }
        passed, reason = _validate_sample(sample, require_context_for_types={"yaml_reference"})
        self.assertFalse(passed)
        self.assertEqual(reason, "missing_context_input")

    def test_char_chunk_sample_splits_long_char_output(self):
        long_line = "x" * 2500
        sample = {
            "instruction": "Show me the implementation of the class Example in the file <source_file>.",
            "input": "Source file: <source_file>",
            "output": "\n".join([long_line, long_line, long_line]),
            "metadata": {"source": "repos/example/src/Example.php", "type": "code_reference"},
        }
        chunks = _char_chunk_sample(sample, max_output_chars=3000, overlap_lines=1)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["output"]), 3000)
            self.assertIn("char_chunk_index", chunk["metadata"]["refinement"])

    def test_deduplicate_normalized_outputs_prefers_augmented(self):
        retrieval = {
            "instruction": "Show me the implementation of the class Example in the file <source_file>.",
            "input": "Source file: <source_file>",
            "output": "<?php\n\nclass Example {}\n",
            "metadata": {"source": "repos/example/src/Example.php", "sample_type": "retrieval"},
        }
        bugfix = {
            "instruction": "Fix this broken Drupal 11 class implementation and return corrected PHP code.",
            "input": "Source file: <source_file>",
            "output": "<?php\nclass Example {}\n",
            "metadata": {
                "source": "repos/example/src/Example.php",
                "sample_type": "bugfix",
                "refinement": {"augmentation_type": "bugfix"},
            },
        }
        kept, dropped = _deduplicate_normalized_outputs(
            [retrieval, bugfix],
            resolution="prefer_augmented_drop_retrieval",
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(kept[0]["metadata"]["sample_type"], "bugfix")


if __name__ == "__main__":
    unittest.main()
