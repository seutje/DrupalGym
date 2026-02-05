import os
import json
import hashlib
import re
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from .manifest import Manifest, calculate_hash
from .logger import PipelineLogger

class Normalizer:
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.seen_hashes = {} # hash -> original_relative_path
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "deduplicated_files": 0,
            "rejected_files": 0,
            "bytes_saved": 0
        }

    def normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        # Standardize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split('\n')]
        # Remove multiple trailing newlines
        normalized = '\n'.join(lines).strip()
        return normalized + '\n'

    def strip_php_license(self, content: str) -> str:
        """Strip common PHP license headers."""
        # Matches common GPL header patterns in DocBlocks
        patterns = [
            r'/\*\*.*?@license.*?GPL.*? \*/'
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        return content

    def clean_html(self, html_content: str, url_hint: str = "") -> str:
        """Extract main content from HTML docs and convert to Markdown."""
        # Suppress the warning by being explicit about the features
        soup = BeautifulSoup(html_content, 'lxml')
        
        # 1. Aggressive Boilerplate Removal
        noise_selectors = [
            'nav', 'footer', 'header', 'aside', 'script', 'style', 'noscript',
            '.region-header', '.region-footer', '.region-sidebar-first', '.region-sidebar-second',
            '.breadcrumb', '.visually-hidden', '#skip-link', '.cookie-banner', 
            '.eu-cookie-compliance-banner', '.messages--warning', '.messages--error',
            '.search-block-form', '.navigation', '.contextual', '.social-media-links',
            '.feedback-link', '#drupal-live-announce', '.field--name-uid', '.field--name-created',
            '#block-api-drupal-org-cookieconsent', '.api-nav-tabs', '#block-bluecheese-branding'
        ]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Specific removal for the annoying cookie text if it's not in a selector we know
        for element in soup.find_all(string=re.compile(r"Can we use first and third party cookies")):
            parent = element.find_parent()
            if parent:
                parent.decompose()

        # 2. Target specific content areas
        content_selectors = [
            'article', 
            '.main-content', 
            '.documentation-content', 
            '#main-content', 
            '.node__content', 
            '.field--name-body', 
            '.api-content',
            '#block-system-main'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup

        # 3. Handle Titles (Ensure we have an H1)
        h1 = soup.find('h1')
        if h1 and not main_content.find('h1'):
            # Re-insert H1 at the top of content if it's missing from the main area
            main_content.insert(0, h1)

        # 4. Filter out old Drupal versions if explicit
        text_full = main_content.get_text().lower()
        if "drupal 7" in text_full and "drupal 11" not in text_full and "drupal 10" not in text_full:
            if "security support for drupal 7 ended" in text_full:
                 # This is the banner we saw in the user's example
                 pass # We already decomposed the banner above hopefully
            
        # 5. Convert to Markdown
        markdown = md(str(main_content), heading_style="ATX")
        
        # 6. Post-processing cleanup
        markdown = re.sub(r'\n\s*\n', '\n\n', markdown) # Collapse multiple newlines
        markdown = re.sub(r'\[Edit\].*?\n', '', markdown) # Remove edit links
        
        return markdown.strip()

    def process_file(self, raw_path: Path, clean_dir: Path, root: Path) -> bool:
        self.stats["total_files"] += 1
        
        # Skip certain binary or irrelevant files
        if raw_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot']:
            return False

        try:
            with open(raw_path, 'rb') as f:
                raw_bytes = f.read()
            
            # Detect if it's likely binary
            if b'\x00' in raw_bytes and raw_path.suffix.lower() not in ['.php', '.module', '.inc']:
                return False

            content = raw_bytes.decode('utf-8', errors='ignore')
            original_size = len(raw_bytes)
            
            # Apply normalization based on file type
            if raw_path.suffix.lower() == '.html':
                content = self.clean_html(content, str(raw_path))
                # Basic quality filter for docs
                if len(content) < 200: # Very short docs are usually just redirects or broken
                    self.stats["rejected_files"] += 1
                    return False
                if "drupal 7" in content.lower() and "drupal 11" not in content.lower() and "drupal 10" not in content.lower():
                     if "benchmarking and profiling drupal" not in content.lower(): # some generic stuff is fine but D7 specific is not
                        self.stats["rejected_files"] += 1
                        return False

            elif raw_path.suffix.lower() in ['.php', '.module', '.inc', '.install', '.profile', '.theme']:
                content = self.strip_php_license(content)
            
            content = self.normalize_text(content)
            
            # Deduplication
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if content_hash in self.seen_hashes:
                self.stats["deduplicated_files"] += 1
                self.stats["bytes_saved"] += original_size
                return False

            # Save clean file
            rel_path = raw_path.relative_to(root / "raw")
            target_path = clean_dir / rel_path
            # Change .html to .md for docs
            if target_path.suffix.lower() == '.html':
                target_path = target_path.with_suffix('.md')
                
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.seen_hashes[content_hash] = str(rel_path)
            self.stats["processed_files"] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error processing {raw_path}: {str(e)}")
            return False

def run_normalization_stage(config: dict, logger: PipelineLogger, root: Path):
    raw_manifest_path = root / "raw" / "manifest.json"
    if not raw_manifest_path.exists():
        logger.error("raw/manifest.json not found. Run acquisition stage first.")
        return 1

    clean_dir = root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = Manifest("normalization", clean_dir)
    manifest.add_input("raw_manifest", "1.0", calculate_hash(raw_manifest_path))

    normalizer = Normalizer(logger)
    
    # Iterate through all files in raw/
    raw_root = root / "raw"
    for dirpath, _, filenames in os.walk(raw_root):
        for filename in filenames:
            raw_path = Path(dirpath) / filename
            normalizer.process_file(raw_path, clean_dir, root)

    manifest.set_metrics(normalizer.stats)
    
    # Save deduplication mapping for traceability
    dedup_manifest_path = clean_dir / "dedup_manifest.json"
    with open(dedup_manifest_path, "w") as f:
        json.dump(normalizer.seen_hashes, f, indent=2)
    
    manifest.add_output("dedup_manifest", "clean/dedup_manifest.json", calculate_hash(dedup_manifest_path))
    manifest.save()
    
    logger.info(f"Normalization complete. Metrics: {normalizer.stats}")
    return 0
