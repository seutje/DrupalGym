import json
import tempfile
import unittest
from pathlib import Path

from drupalgym.discovery import DiscoveryConfig, _constraint_matches, discover_sources


class DiscoveryTests(unittest.TestCase):
    def test_constraint_matches(self) -> None:
        self.assertTrue(_constraint_matches("^11", "^11"))
        self.assertTrue(_constraint_matches(">=11", "^11"))
        self.assertTrue(_constraint_matches("11.0", "^11"))
        self.assertFalse(_constraint_matches("^10", "^11"))

    def test_discover_sources_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest = tmp_path / "sources" / "manifest.json"
            config = DiscoveryConfig(
                output_manifest=manifest,
                drupal_core_repo="https://example.com/drupal.git",
                project_api_urls=[],
                doc_urls=["https://example.com/docs"],
            )

            entries = discover_sources(config)

            self.assertTrue(manifest.exists())
            data = json.loads(manifest.read_text())
            self.assertEqual(data["core_constraint"], "^11")
            self.assertEqual(entries[0].name, "drupal-core")


if __name__ == "__main__":
    unittest.main()
