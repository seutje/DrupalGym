import tempfile
import unittest
from pathlib import Path

from drupalgym.dedup import DedupConfig, deduplicate_sources
from drupalgym.normalize import NormalizeConfig, normalize_sources


class NormalizeDedupTests(unittest.TestCase):
    def test_normalize_and_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            clean_root = tmp_path / "clean"
            dedup_root = tmp_path / "dedup"

            php_file = raw_root / "code" / "example.php"
            php_file.parent.mkdir(parents=True)
            php_file.write_text(
                """/* License: MIT */\n\n<?php\n\n$var = 'value';\n""",
                encoding="utf-8",
            )
            duplicate = raw_root / "code" / "duplicate.php"
            duplicate.write_text(
                php_file.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            normalize_sources(
                NormalizeConfig(
                    raw_root=raw_root,
                    clean_root=clean_root,
                    manifest_path=clean_root / "manifest.json",
                )
            )

            output = deduplicate_sources(
                DedupConfig(
                    clean_root=clean_root,
                    dedup_root=dedup_root,
                    manifest_path=clean_root / "dedup_manifest.json",
                    ignore_paths=(clean_root / "manifest.json",),
                )
            )

            self.assertEqual(len(output), 1)
            example_exists = (dedup_root / "code" / "example.php").exists()
            duplicate_exists = (dedup_root / "code" / "duplicate.php").exists()
            self.assertNotEqual(example_exists, duplicate_exists)


if __name__ == "__main__":
    unittest.main()
