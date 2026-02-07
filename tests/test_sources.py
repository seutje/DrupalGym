import unittest

from pipeline.sources import (
    _extract_default_branch,
    _extract_machine_name,
    _is_drupal_core_11,
    _supports_php_min,
)


class SourcesHelpersTest(unittest.TestCase):
    def test_is_drupal_core_11(self):
        self.assertTrue(_is_drupal_core_11("^11"))
        self.assertTrue(_is_drupal_core_11("11.x-dev"))
        self.assertFalse(_is_drupal_core_11("^10"))

    def test_extract_machine_name(self):
        node = {"field_project_machine_name": {"value": "example_project"}}
        self.assertEqual(_extract_machine_name(node), "example_project")

    def test_extract_default_branch(self):
        node = {"field_project_default_branch": {"value": "11.x"}}
        self.assertEqual(_extract_default_branch(node), "11.x")

    def test_supports_php_min(self):
        self.assertTrue(_supports_php_min("^8.3 || ^8.4", "8.3"))
        self.assertFalse(_supports_php_min("^8.1 || ^8.2", "8.3"))


if __name__ == "__main__":
    unittest.main()
