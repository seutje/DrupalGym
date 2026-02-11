import unittest

from pipeline.sft_generation import InstructionGenerator


class _DummyLogger:
    def error(self, *_args, **_kwargs):
        return None


class SftGenerationHelpersTest(unittest.TestCase):
    def setUp(self):
        self.generator = InstructionGenerator(
            _DummyLogger(),
            config={
                "enable_symbol_kind_prompts": True,
                "doc_source_allowlist_prefixes": ["docs/www_drupal_org/", "docs/api_drupal_org/"],
                "doc_topic_denylist_terms": ["mcp", "apidog"],
            },
        )

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
        self.assertEqual(sample["metadata"]["sample_type"], "retrieval")
        self.assertIn("interface EntityDescriptionInterface", sample["instruction"])
        self.assertNotIn("repos/", sample["instruction"])
        self.assertIn("<source_file>", sample["instruction"])
        self.assertIn("Source file: <source_file>", sample["input"])
        self.assertIn("Namespace:", sample["input"])

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
        self.assertIn("File name hint:", sample["input"])

    def test_yaml_instruction_is_retrieval_only(self):
        content = "services:\n  gym.logger: {}\n"
        self.generator.generate_from_yaml(content, "repos/example/gym.services.yml")
        sample = self.generator.samples[0]
        self.assertEqual(
            sample["instruction"],
            "Provide the Drupal 11 YAML configuration from <source_file>. Return one fenced yaml block only.",
        )
        self.assertNotIn("repos/", sample["instruction"])
        self.assertNotIn("explain what it defines", sample["instruction"].lower())
        self.assertIn("Source file: <source_file>", sample["input"])
        self.assertIn("Configuration topic:", sample["input"])
        self.assertEqual(sample["metadata"]["sample_type"], "retrieval")

    def test_doc_generation_honors_allowlist_and_denylist(self):
        content = "# State API\n\nDrupal 11 State API stores ephemeral data for services and runtime state.\n\nUse it for non-content state that is environment specific.\n"
        self.generator.generate_from_doc(content, "docs/www_drupal_org/docs/state-api.md")
        self.assertEqual(len(self.generator.samples), 1)
        sample = self.generator.samples[0]
        self.assertIn("Explain the following topic", sample["instruction"])
        self.assertIn("Source file: <source_file>", sample["input"])
        self.assertIn("Topic: State API", sample["output"])
        self.assertEqual(sample["metadata"]["sample_type"], "retrieval")

        blocked_content = "# MCP Client\n\nThis page explains MCP workflows."
        self.generator.generate_from_doc(blocked_content, "docs/www_drupal_org/docs/mcp-client.md")
        self.assertEqual(len(self.generator.samples), 1)

    def test_generate_sdc_bundle(self):
        self.generator.generate_sdc_bundle(
            yaml_rel_path="repos/example/components/card/card.component.yml",
            yaml_content="name: Card\nprops:\n  title:\n    type: string\n",
            twig_rel_path="repos/example/components/card/card.twig",
            twig_content="<article>{{ title }}</article>\n",
        )
        self.assertEqual(len(self.generator.samples), 4)
        sample = self.generator.samples[0]
        self.assertEqual(sample["metadata"]["type"], "sdc_reference")
        self.assertEqual(sample["metadata"]["sample_type"], "retrieval")
        self.assertIn("Single Directory Component", sample["instruction"])
        self.assertIn("component.yml", sample["output"])
        self.assertIn("card.twig", sample["output"])
        self.assertIn("Component name: card", sample["input"])
        usage_variant = self.generator.samples[2]
        self.assertIn("Example usage", usage_variant["output"])
        contract_variant = self.generator.samples[3]
        self.assertIn("Component contract checklist", contract_variant["output"])

    def test_generate_change_record_refactor_samples(self):
        content = (
            "# Convert hook_help to service class\n\n"
            "## Change Record Metadata\n"
            "- Status: published\n"
            "- Target versions: 10.3.x, 11.x\n"
            "- Updated: 2025-10-01T10:00:00Z\n"
            "- URL: https://www.drupal.org/node/123456\n\n"
            "## Rationale\n"
            "Drupal 11 moves this behavior to a service for lazy loading and cleaner separation.\n\n"
            "## Before\n"
            "```php\n"
            "<?php\n"
            "function example_help() {\n"
            "  return 'legacy';\n"
            "}\n"
            "```\n\n"
            "## After\n"
            "```php\n"
            "<?php\n"
            "final class ExampleHelpBuilder {\n"
            "  public function build(): string {\n"
            "    return 'modern';\n"
            "  }\n"
            "}\n"
            "```\n"
        )
        self.generator.generate_from_doc(content, "docs/www_drupal_org/change_records/123456.md")
        self.assertEqual(len(self.generator.samples), 1)
        sample = self.generator.samples[0]
        self.assertEqual(sample["metadata"]["type"], "refactor")
        self.assertIn("Refactor this Drupal code", sample["instruction"])
        self.assertIn("Legacy code:", sample["input"])
        self.assertIn("Change record title: Convert hook_help to service class", sample["input"])
        self.assertIn("final class ExampleHelpBuilder", sample["output"])


if __name__ == "__main__":
    unittest.main()
