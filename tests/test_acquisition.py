import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.acquisition import (
    _coerce_bool,
    _count_files_and_bytes,
    _doc_fetch_is_valid,
    _doc_path_for_url,
    _parse_change_record_page,
)


class AcquisitionHelpersTest(unittest.TestCase):
    def test_coerce_bool_parses_common_values(self):
        self.assertTrue(_coerce_bool(True, default=False))
        self.assertTrue(_coerce_bool("true", default=False))
        self.assertTrue(_coerce_bool("YES", default=False))
        self.assertFalse(_coerce_bool("false", default=True))
        self.assertFalse(_coerce_bool(0, default=True))
        self.assertTrue(_coerce_bool(None, default=True))
        self.assertFalse(_coerce_bool("unexpected", default=False))

    def test_doc_path_for_url_maps_to_expected_cache_location(self):
        root = Path("/tmp/raw_docs")
        self.assertEqual(
            _doc_path_for_url(root, "https://www.drupal.org/docs/develop/theming"),
            root / "www_drupal_org" / "docs/develop/theming.html",
        )
        self.assertEqual(
            _doc_path_for_url(root, "https://symfony.com/doc/7.0/index.html"),
            root / "symfony_com" / "doc/7.0/index.html",
        )

    def test_count_files_and_bytes_supports_suffix_filter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").mkdir(parents=True, exist_ok=True)
            (root / "a" / "one.md").write_text("abc", encoding="utf-8")
            (root / "a" / "two.txt").write_text("12345", encoding="utf-8")
            (root / "a" / "three.md").write_text("z", encoding="utf-8")

            all_count, all_bytes = _count_files_and_bytes(root)
            self.assertEqual(all_count, 3)
            self.assertEqual(all_bytes, 9)

            md_count, md_bytes = _count_files_and_bytes(root, suffixes=(".md",))
            self.assertEqual(md_count, 2)
            self.assertEqual(md_bytes, 4)

    def test_doc_fetch_valid_requires_success_and_pages(self):
        self.assertTrue(_doc_fetch_is_valid({"success": True, "pages": 3}))
        self.assertFalse(_doc_fetch_is_valid({"success": True, "pages": 0}))
        self.assertFalse(_doc_fetch_is_valid({"success": False, "pages": 10}))

    def test_parse_change_record_page_accepts_published_with_before_after(self):
        html = """
        <html>
          <body>
            <h1>Convert legacy API usage</h1>
            <div class="field field--name-field-change-record-status">Published</div>
            <div class="field field--name-field-affected-version">11.x, 10.3.x</div>
            <time datetime="2025-12-15T10:00:00Z">Dec 15, 2025</time>
            <div class="change-record-description">
              <p>This change updates service construction patterns for Drupal 11.</p>
              <h2>Before</h2>
              <pre><code class="language-php">&lt;?php
              $service = \\Drupal::service('logger.factory');
              </code></pre>
              <h2>After</h2>
              <pre><code class="language-php">&lt;?php
              final class Example {
                public function __construct(private LoggerChannelFactoryInterface $factory) {}
              }
              </code></pre>
            </div>
          </body>
        </html>
        """
        parsed = _parse_change_record_page(
            html,
            "https://www.drupal.org/node/123456",
            target_versions={"11.x", "10.3.x"},
            lookback_cutoff=datetime.now(timezone.utc) - timedelta(days=365 * 2),
        )
        self.assertTrue(parsed["accepted"])
        self.assertIn("## Before", parsed["markdown"])
        self.assertIn("## After", parsed["markdown"])
        self.assertEqual(parsed["node_id"], "123456")

    def test_parse_change_record_page_rejects_draft(self):
        html = """
        <html>
          <body>
            <h1>Draft change</h1>
            <div class="field field--name-field-change-record-status">Draft</div>
            <div class="change-record-description">
              <h2>Before</h2>
              <pre><code>old()</code></pre>
              <h2>After</h2>
              <pre><code>new()</code></pre>
            </div>
          </body>
        </html>
        """
        parsed = _parse_change_record_page(
            html,
            "https://www.drupal.org/node/1",
            target_versions={"11.x"},
            lookback_cutoff=None,
        )
        self.assertFalse(parsed["accepted"])
        self.assertEqual(parsed["reason"], "draft")

    def test_parse_change_record_page_rejects_version_outside_target(self):
        html = """
        <html>
          <body>
            <h1>Only Drupal 10.1</h1>
            <div class="field field--name-field-change-record-status">Published</div>
            <div class="field field--name-field-affected-version">10.1.x</div>
            <div class="change-record-description">
              <h2>Before</h2>
              <pre><code>old()</code></pre>
              <h2>After</h2>
              <pre><code>new()</code></pre>
            </div>
          </body>
        </html>
        """
        parsed = _parse_change_record_page(
            html,
            "https://www.drupal.org/node/2",
            target_versions={"11.x", "10.3.x"},
            lookback_cutoff=None,
        )
        self.assertFalse(parsed["accepted"])
        self.assertEqual(parsed["reason"], "version_filtered")


if __name__ == "__main__":
    unittest.main()
