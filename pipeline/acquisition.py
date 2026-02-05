import json
import subprocess
import requests
import hashlib
import time
import re
from pathlib import Path
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from .manifest import Manifest, calculate_hash
from .logger import PipelineLogger

class DocumentationFetcher:
    def __init__(self, logger: PipelineLogger, base_docs_dir: Path):
        self.logger = logger
        self.base_docs_dir = base_docs_dir
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DrupalGym/1.0 (Training Pipeline)"})

    def fetch_url(self, url: str, target_file: Path):
        if target_file.exists():
            return True

        self.logger.info(f"Fetching {url}")
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            with open(target_file, "wb") as f:
                f.write(response.content)
            time.sleep(0.5)  # Politeness delay
            return True
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {str(e)}")
            return False

    def get_doc_path(self, url: str) -> Path:
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_")
        path = parsed.path.strip("/")
        if not path:
            path = "index"
        if not path.endswith((".html", ".md", ".json")):
            path += ".html"
        return self.base_docs_dir / domain / path

    def discover_links(self, url: str, base_prefix: str):
        """Discovers links within the same base prefix."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            links = []
            for a in soup.find_all('a', href=True):
                full_url = urljoin(url, a['href']).split('#')[0]
                if full_url.startswith(base_prefix) and full_url not in self.visited:
                    links.append(full_url)
            return links
        except Exception as e:
            self.logger.error(f"Discovery failed for {url}: {e}")
            return []

    def recursive_fetch(self, start_url: str, base_prefix: str, max_pages: int = 100):
        queue = [start_url]
        count = 0
        
        while queue and count < max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
            
            self.visited.add(url)
            target_file = self.get_doc_path(url)
            
            if self.fetch_url(url, target_file):
                count += 1
                # Discover more if it's an HTML page
                if url.endswith((".html", "/")) or "." not in url.split("/")[-1] or "api.drupal.org" in url:
                    new_links = self.discover_links(url, base_prefix)
                    queue.extend([l for l in new_links if l not in self.visited])
            
        self.logger.info(f"Finished recursive fetch. Captured {count} pages.")
        return count

def clone_or_fetch(url: str, ref: str, target_dir: Path, logger: PipelineLogger):
    if target_dir.exists():
        logger.info(f"Fetching updates for {url}")
        try:
            subprocess.run(["git", "fetch", "--all"], cwd=target_dir, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch {url}: {e.stderr.decode()}")
            return False
    else:
        logger.info(f"Cloning {url}")
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", url, str(target_dir)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {url}: {e.stderr.decode()}")
            return False

    try:
        subprocess.run(["git", "checkout", ref], cwd=target_dir, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to checkout {ref}: {e.stderr.decode()}")
        return False

def run_acquisition_stage(config: dict, logger: PipelineLogger, root: Path):
    sources_manifest_path = root / "sources" / "manifest.json"
    if not sources_manifest_path.exists():
        logger.error("sources/manifest.json not found.")
        return 1

    with open(sources_manifest_path, "r") as f:
        sources_data = json.load(f)

    raw_dir = root / "raw"
    repos_dir = raw_dir / "repos"
    docs_dir = raw_dir / "docs"
    
    manifest = Manifest("acquisition", raw_dir)
    manifest.add_input("sources_manifest", "1.0", calculate_hash(sources_manifest_path))

    fetcher = DocumentationFetcher(logger, docs_dir)
    success_count = 0
    failure_count = 0
    doc_pages_total = 0

    # Process curated sources
    for source in sources_data.get("sources", {}).get("curated", []):
        source_id = source["id"]
        source_type = source["type"]
        url = source["url"]
        ref = source.get("ref")

        if source_type == "git":
            target_dir = repos_dir / source_id
            if clone_or_fetch(url, ref, target_dir, logger):
                commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target_dir).decode().strip()
                manifest.add_output(source_id, str(target_dir.relative_to(root)), commit_hash)
                success_count += 1
            else:
                failure_count += 1
        elif source_type == "http":
            # For specific documentation sources, use recursive fetch
            # We define prefix rules here
            prefix = url
            if "symfony.com" in url:
                prefix = "https://symfony.com/doc/7.0/"
            elif "drupal.org/docs" in url:
                prefix = "https://www.drupal.org/docs/develop" # Targeted crawl
            
            logger.info(f"Starting recursive fetch for {source_id} with prefix {prefix}")
            pages = fetcher.recursive_fetch(url, prefix, max_pages=200) # Limit for now
            doc_pages_total += pages
            manifest.add_output(source_id, str((docs_dir / urlparse(url).netloc.replace(".", "_")).relative_to(root)), "collection")
            success_count += 1

    # Process discovered projects
    for source in sources_data.get("sources", {}).get("drupal_org_projects", []):
        source_id = source["id"]
        url = source["url"]
        ref = source.get("ref", "master")
        target_dir = repos_dir / source_id
        if clone_or_fetch(url, ref, target_dir, logger):
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target_dir).decode().strip()
            manifest.add_output(source_id, str(target_dir.relative_to(root)), commit_hash)
            success_count += 1
        else:
            failure_count += 1

    manifest.set_metrics({
        "success_count": success_count,
        "failure_count": failure_count,
        "doc_pages_captured": doc_pages_total,
        "total_repos": len(sources_data.get("sources", {}).get("drupal_org_projects", [])) + 1
    })
    
    manifest.save()
    return 0 if failure_count == 0 else 1
