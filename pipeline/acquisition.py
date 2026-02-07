import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash


class DocumentationFetcher:
    def __init__(self, logger: PipelineLogger, base_docs_dir: Path):
        self.logger = logger
        self.base_docs_dir = base_docs_dir
        self.visited = set()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DrupalGym/1.0 (Training Pipeline)"})

    def fetch_url(self, url: str, target_file: Path) -> dict:
        if target_file.exists():
            return {
                "success": True,
                "status": 200,
                "bytes": target_file.stat().st_size,
                "retried": 0,
                "cached": True,
            }

        retries = 2
        for attempt in range(retries + 1):
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                response = self.session.get(url, timeout=30)
                status_code = response.status_code
                response.raise_for_status()
                with open(target_file, "wb") as handle:
                    handle.write(response.content)
                time.sleep(0.3)
                return {
                    "success": True,
                    "status": status_code,
                    "bytes": len(response.content),
                    "retried": attempt,
                    "cached": False,
                }
            except Exception as exc:
                if attempt >= retries:
                    self.logger.error(f"Failed to fetch {url}: {str(exc)}")
                    return {
                        "success": False,
                        "status": None,
                        "bytes": 0,
                        "retried": attempt,
                        "cached": False,
                    }
                time.sleep(0.4 * (attempt + 1))
        return {
            "success": False,
            "status": None,
            "bytes": 0,
            "retried": retries,
            "cached": False,
        }

    def get_doc_path(self, url: str) -> Path:
        parsed = urlparse(url)
        domain = parsed.netloc.replace(".", "_")
        path = parsed.path.strip("/")
        if not path:
            path = "index"
        if not path.endswith((".html", ".md", ".json", ".xml")):
            path += ".html"
        return self.base_docs_dir / domain / path

    def discover_links(self, url: str, allowed_prefixes: list[str]) -> list[str]:
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            links: list[str] = []
            for anchor in soup.find_all("a", href=True):
                full_url = urljoin(url, anchor["href"]).split("#")[0]
                if full_url in self.visited:
                    continue
                if any(full_url.startswith(prefix) for prefix in allowed_prefixes):
                    links.append(full_url)
            return links
        except Exception as exc:
            self.logger.error(f"Discovery failed for {url}: {exc}")
            return []

    def recursive_fetch(self, start_url: str, allowed_prefixes: list[str], max_pages: int = 100) -> dict:
        queue = [start_url]
        captured = 0
        bytes_written = 0
        retries_total = 0
        failed_pages = 0

        while queue and captured < max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
            if not any(url.startswith(prefix) for prefix in allowed_prefixes):
                continue

            self.visited.add(url)
            target_file = self.get_doc_path(url)
            result = self.fetch_url(url, target_file)
            retries_total += int(result.get("retried", 0))

            if result.get("success"):
                captured += 1
                bytes_written += int(result.get("bytes", 0))
                if url.endswith((".html", "/")) or "." not in url.split("/")[-1] or "api.drupal.org" in url:
                    queue.extend(link for link in self.discover_links(url, allowed_prefixes) if link not in self.visited)
            else:
                failed_pages += 1

        self.logger.info(f"Finished recursive fetch. Captured {captured} pages.")
        return {
            "success": failed_pages == 0,
            "pages": captured,
            "bytes": bytes_written,
            "retried": retries_total,
            "failed_pages": failed_pages,
        }


def clone_or_fetch(url: str, ref: str, target_dir: Path, logger: PipelineLogger) -> dict:
    action = "fetch" if target_dir.exists() else "clone"
    retried = 0

    if target_dir.exists():
        logger.info(f"Fetching updates for {url}")
        try:
            subprocess.run(["git", "fetch", "--all"], cwd=target_dir, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.error(f"Failed to fetch {url}: {exc.stderr.decode(errors='ignore')}")
            return {"success": False, "action": action, "retried": retried, "commit": None}
    else:
        logger.info(f"Cloning {url}")
        try:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", url, str(target_dir)], check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.error(f"Failed to clone {url}: {exc.stderr.decode(errors='ignore')}")
            return {"success": False, "action": action, "retried": retried, "commit": None}

    try:
        subprocess.run(["git", "checkout", ref], cwd=target_dir, check=True, capture_output=True)
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target_dir).decode().strip()
        return {
            "success": True,
            "action": action,
            "retried": retried,
            "commit": commit_hash,
        }
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to checkout {ref}: {exc.stderr.decode(errors='ignore')}")
        return {"success": False, "action": action, "retried": retried, "commit": None}


def _default_prefix_for_url(url: str) -> list[str]:
    if "symfony.com" in url:
        return ["https://symfony.com/doc/7.0/"]
    if "drupal.org/docs" in url:
        return ["https://www.drupal.org/docs/develop"]
    if "api.drupal.org" in url:
        return ["https://api.drupal.org/api/drupal"]
    if "drupal.org/security" in url:
        return ["https://www.drupal.org/security"]
    return [url]


def run_acquisition_stage(config: dict, logger: PipelineLogger, root: Path):
    sources_manifest_path = root / "sources" / "manifest.json"
    if not sources_manifest_path.exists():
        logger.error("sources/manifest.json not found.")
        return 1

    with open(sources_manifest_path, "r", encoding="utf-8") as handle:
        sources_data = json.load(handle)

    raw_dir = root / "raw"
    repos_dir = raw_dir / "repos"
    docs_dir = raw_dir / "docs"

    manifest = Manifest("acquisition", raw_dir)
    manifest.add_input("sources_manifest", "1.0", calculate_hash(sources_manifest_path))

    fetcher = DocumentationFetcher(logger, docs_dir)
    success_count = 0
    failure_count = 0
    doc_pages_total = 0
    fetch_status: list[dict] = []

    docs_cfg = config.get("acquisition", {}).get("docs", {})
    max_pages_per_source = int(docs_cfg.get("max_pages_per_source", 200))
    allowed_prefixes_cfg = docs_cfg.get("allowed_prefixes", {})

    for source in sources_data.get("sources", {}).get("curated", []):
        source_id = source["id"]
        source_type = source["type"]
        url = source["url"]
        ref = source.get("ref")

        if source_type == "git":
            target_dir = repos_dir / source_id
            result = clone_or_fetch(url, ref, target_dir, logger)
            if result.get("success"):
                manifest.add_output(source_id, str(target_dir.relative_to(root)), str(result.get("commit")))
                success_count += 1
            else:
                failure_count += 1
            fetch_status.append(
                {
                    "source_id": source_id,
                    "type": "git",
                    "success": bool(result.get("success")),
                    "status": "ok" if result.get("success") else "failed",
                    "bytes": 0,
                    "retried": int(result.get("retried", 0)),
                    "action": result.get("action"),
                    "commit": result.get("commit"),
                }
            )
            continue

        if source_type == "http":
            allowed_prefixes = allowed_prefixes_cfg.get(source_id)
            if not isinstance(allowed_prefixes, list) or not allowed_prefixes:
                allowed_prefixes = _default_prefix_for_url(url)

            logger.info(
                f"Starting recursive fetch for {source_id}",
                source_id=source_id,
                allowed_prefixes=allowed_prefixes,
                max_pages=max_pages_per_source,
            )
            result = fetcher.recursive_fetch(url, allowed_prefixes=allowed_prefixes, max_pages=max_pages_per_source)
            doc_pages_total += int(result.get("pages", 0))
            manifest.add_output(
                source_id,
                str((docs_dir / urlparse(url).netloc.replace(".", "_")).relative_to(root)),
                "collection",
            )
            if result.get("success"):
                success_count += 1
            else:
                failure_count += 1

            fetch_status.append(
                {
                    "source_id": source_id,
                    "type": "http",
                    "success": bool(result.get("success")),
                    "status": "ok" if result.get("success") else "failed",
                    "bytes": int(result.get("bytes", 0)),
                    "retried": int(result.get("retried", 0)),
                    "pages": int(result.get("pages", 0)),
                    "failed_pages": int(result.get("failed_pages", 0)),
                    "allowed_prefixes": allowed_prefixes,
                }
            )

    for source in sources_data.get("sources", {}).get("drupal_org_projects", []):
        source_id = source["id"]
        url = source["url"]
        ref = source.get("ref", "master")
        target_dir = repos_dir / source_id
        result = clone_or_fetch(url, ref, target_dir, logger)
        if result.get("success"):
            manifest.add_output(source_id, str(target_dir.relative_to(root)), str(result.get("commit")))
            success_count += 1
        else:
            failure_count += 1

        fetch_status.append(
            {
                "source_id": source_id,
                "type": "git",
                "success": bool(result.get("success")),
                "status": "ok" if result.get("success") else "failed",
                "bytes": 0,
                "retried": int(result.get("retried", 0)),
                "action": result.get("action"),
                "commit": result.get("commit"),
            }
        )

    manifest.data["fetch_status"] = fetch_status
    manifest.set_metrics(
        {
            "success_count": success_count,
            "failure_count": failure_count,
            "doc_pages_captured": doc_pages_total,
            "total_repos": len(sources_data.get("sources", {}).get("drupal_org_projects", [])) + 1,
        }
    )

    manifest.save()
    return 0 if failure_count == 0 else 1
