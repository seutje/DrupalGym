import os
import json
import hashlib
import re
from pathlib import Path
from bs4 import BeautifulSoup
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

    def clean_html(self, html_content: str) -> str:
        """Extract main content from HTML docs."""
        # Suppress the warning by being explicit about the features
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove common boilerplate elements
        for tag in ['nav', 'footer', 'header', 'aside', 'script', 'style', 'noscript']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove elements by common classes/ids used for boilerplate
        boilerplate_selectors = [
            '.region-header', '.region-footer', '.breadcrumb', '.visually-hidden',
            '#skip-link', '.cookie-banner', '.search-block-form', '.navigation'
        ]
        for selector in boilerplate_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Target specific content areas if known (Drupal.org, Symfony)
        content_selectors = [
            'article', '.main-content', '.documentation-content', '#main-content', 
            '.node__content', '.field--name-body', '.api-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text and clean up multiple newlines
        text = main_content.get_text(separator='\n')
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def process_file(self, raw_path: Path, clean_dir: Path, root: Path) -> bool:
        self.stats["total_files"] += 1
        
        # Skip certain binary or irrelevant files
        if raw_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot']:
            return False

        try:
            with open(raw_path, 'rb') as f:
                raw_bytes = f.read()
            
            # Detect if it's likely binary
            if b'\x00' in raw_bytes:
                return False

            content = raw_bytes.decode('utf-8', errors='ignore')
            original_size = len(raw_bytes)
            
            # Apply normalization based on file type
            if raw_path.suffix.lower() == '.html':
                content = self.clean_html(content)
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

    with open(raw_manifest_path, "r") as f:
        raw_data = json.load(f)

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
