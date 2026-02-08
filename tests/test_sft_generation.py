import unittest

from pipeline.sft_generation import InstructionGenerator


class _DummyLogger:
    def error(self, *_args, **_kwargs):
        return None


class SftGenerationHelpersTest(unittest.TestCase):
    def setUp(self):
        self.generator = InstructionGenerator(_DummyLogger(), config={"enable_symbol_kind_prompts": True})

    def test_extracts_symbol_from_declaration_not_comment(self):
        content = (
            "<?php\n"
            "/**\n"
            " * Defines the interface for entities that have a description.\n"
            " */\n"
            "interface EntityDescriptionInterface {}\n"
        )
        self.generator.generate_from_php(content, "repos/drupal_core/core/lib/Drupal/Core/Entity/EntityDescriptionInterface.php")
        sample = self.generator.samples[0]
        self.assertEqual(sample["metadata"]["symbol_kind"], "interface")
        self.assertEqual(sample["metadata"]["symbol_name"], "EntityDescriptionInterface")
        self.assertEqual(sample["metadata"]["symbol_extraction_method"], "declaration")
        self.assertIn("interface EntityDescriptionInterface", sample["instruction"])
        self.assertNotIn("repos/", sample["instruction"])
        self.assertIn("<source_file>", sample["instruction"])

    def test_multiple_declarations_use_path_fallback(self):
        content = (
            "<?php\n"
            "class FirstThing {}\n"
            "class SecondThing {}\n"
        )
        self.generator.generate_from_php(content, "repos/example/src/multi_thing.php")
        sample = self.generator.samples[0]
        self.assertEqual(sample["metadata"]["symbol_extraction_method"], "path_fallback")
        self.assertEqual(sample["metadata"]["symbol_name"], "MultiThing")
        self.assertIn("class MultiThing", sample["instruction"])
        self.assertNotIn("repos/", sample["instruction"])
        self.assertIn("<source_file>", sample["instruction"])

    def test_yaml_instruction_is_retrieval_only(self):
        content = "services:\n  gym.logger: {}\n"
        self.generator.generate_from_yaml(content, "repos/example/gym.services.yml")
        sample = self.generator.samples[0]
        self.assertEqual(
            sample["instruction"],
            "Provide the Drupal 11 YAML configuration from <source_file>.",
        )
        self.assertNotIn("repos/", sample["instruction"])
        self.assertNotIn("explain what it defines", sample["instruction"].lower())


if __name__ == "__main__":
    unittest.main()
