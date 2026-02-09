import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash

CHANGE_RECORD_NODE_RE = re.compile(r"/node/(\d+)")
CHANGE_RECORD_VERSION_RE = re.compile(r"\b(10\.2\.x|10\.3\.x|11(?:\.[0-9]+)?\.x(?:-dev)?)\b", re.IGNORECASE)
BEFORE_HINT_RE = re.compile(r"\b(before|legacy|deprecated|old)\b", re.IGNORECASE)
AFTER_HINT_RE = re.compile(r"\b(after|new|replacement|updated)\b", re.IGNORECASE)


def _clamp_workers(value: Any, *, default: int, hard_max: int = 16) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(hard_max, parsed))


class DocumentationFetcher:
    def __init__(
        self,
        logger: PipelineLogger,
        base_docs_dir: Path,
        *,
        request_timeout_seconds: int = 30,
        request_delay_seconds: float = 0.3,
        retry_backoff_seconds: float = 0.4,
        max_retries: int = 2,
    ):
        self.logger = logger
        self.base_docs_dir = base_docs_dir
        self.visited = set()
        self.request_timeout_seconds = max(1, int(request_timeout_seconds))
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.max_retries = max(0, int(max_retries))
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
                "text": None,
            }

        retries = self.max_retries
        for attempt in range(retries + 1):
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                response = self.session.get(url, timeout=self.request_timeout_seconds)
                status_code = response.status_code
                response.raise_for_status()
                with open(target_file, "wb") as handle:
                    handle.write(response.content)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                content_type = str(response.headers.get("Content-Type", "")).lower()
                page_text = response.text if "text/" in content_type or "application/xhtml" in content_type else None
                return {
                    "success": True,
                    "status": status_code,
                    "bytes": len(response.content),
                    "retried": attempt,
                    "cached": False,
                    "text": page_text,
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
                        "text": None,
                    }
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        return {
            "success": False,
            "status": None,
            "bytes": 0,
            "retried": retries,
            "cached": False,
            "text": None,
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

    def discover_links(self, url: str, allowed_prefixes: list[str], url_denylist_terms: list[str] | None = None) -> list[str]:
        try:
            response = self.session.get(url, timeout=self.request_timeout_seconds)
            response.raise_for_status()
            if self.request_delay_seconds > 0:
                time.sleep(self.request_delay_seconds)
            return self.discover_links_from_html(response.text, url, allowed_prefixes, url_denylist_terms=url_denylist_terms)
        except Exception as exc:
            self.logger.error(f"Discovery failed for {url}: {exc}")
            return []

    def discover_links_from_html(
        self,
        html: str,
        base_url: str,
        allowed_prefixes: list[str],
        url_denylist_terms: list[str] | None = None,
    ) -> list[str]:
        soup = BeautifulSoup(html or "", "lxml")
        links: list[str] = []
        denylist = [term.lower() for term in (url_denylist_terms or []) if str(term).strip()]
        for anchor in soup.find_all("a", href=True):
            full_url = urljoin(base_url, anchor["href"]).split("#")[0]
            if full_url in self.visited:
                continue
            if any(full_url.startswith(prefix) for prefix in allowed_prefixes):
                full_url_lower = full_url.lower()
                if any(term in full_url_lower for term in denylist):
                    continue
                links.append(full_url)
        return links

    def recursive_fetch(
        self,
        start_url: str,
        allowed_prefixes: list[str],
        max_pages: int = 100,
        url_denylist_terms: list[str] | None = None,
    ) -> dict:
        queue = [start_url]
        captured = 0
        bytes_written = 0
        retries_total = 0
        failed_pages = 0
        denylist = [term.lower() for term in (url_denylist_terms or []) if str(term).strip()]

        while queue and captured < max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
            if not any(url.startswith(prefix) for prefix in allowed_prefixes):
                continue
            if any(term in url.lower() for term in denylist):
                continue

            self.visited.add(url)
            target_file = self.get_doc_path(url)
            result = self.fetch_url(url, target_file)
            retries_total += int(result.get("retried", 0))

            if result.get("success"):
                captured += 1
                bytes_written += int(result.get("bytes", 0))
                if url.endswith((".html", "/")) or "." not in url.split("/")[-1] or "api.drupal.org" in url:
                    discovered = []
                    page_text = result.get("text")
                    if isinstance(page_text, str) and page_text.strip():
                        discovered = self.discover_links_from_html(
                            page_text,
                            url,
                            allowed_prefixes,
                            url_denylist_terms=denylist,
                        )
                    else:
                        discovered = self.discover_links(url, allowed_prefixes, url_denylist_terms=denylist)
                    queue.extend(link for link in discovered if link not in self.visited)
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


def _doc_fetch_is_valid(result: dict) -> bool:
    return bool(result.get("success")) and int(result.get("pages", 0)) > 0


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def _extract_node_id(url: str) -> str:
    match = CHANGE_RECORD_NODE_RE.search(url or "")
    if match:
        return match.group(1)
    return "unknown"


def _safe_filename_fragment(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned[:80] if cleaned else "change-record"


def _parse_datetime_value(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = str(raw).strip()
    if not value:
        return None

    if re.fullmatch(r"\d{10,}", value):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    formats = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _canonicalize_change_record_version(raw: str) -> str:
    value = str(raw).lower()
    if value.startswith("11"):
        return "11.x"
    if value.startswith("10.2"):
        return "10.2.x"
    if value.startswith("10.3"):
        return "10.3.x"
    return raw


def _extract_change_record_versions(field_values: dict[str, str], text_blob: str) -> list[str]:
    candidate_chunks: list[str] = [text_blob]
    for key, value in field_values.items():
        key_lower = key.lower()
        if "version" in key_lower or "branch" in key_lower:
            candidate_chunks.append(value)

    versions: set[str] = set()
    for chunk in candidate_chunks:
        for match in CHANGE_RECORD_VERSION_RE.findall(chunk or ""):
            versions.add(_canonicalize_change_record_version(match))
    return sorted(versions)


def _extract_change_record_fields(soup: BeautifulSoup) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in soup.select("div[class*='field--name-']"):
        classes = field.get("class") or []
        names = [item[len("field--name-") :] for item in classes if item.startswith("field--name-")]
        if not names:
            continue
        text = _normalize_ws(field.get_text(" ", strip=True))
        if not text:
            continue
        for name in names:
            previous = fields.get(name, "")
            fields[name] = _normalize_ws(f"{previous} {text}")
    return fields


def _extract_change_record_status(field_values: dict[str, str], text_blob: str) -> str:
    chunks: list[str] = [text_blob]
    for key, value in field_values.items():
        key_lower = key.lower()
        if "status" in key_lower or "moderation" in key_lower or "state" in key_lower:
            chunks.append(value)

    blob = "\n".join(chunks).lower()
    if "draft" in blob:
        return "draft"
    if "published" in blob:
        return "published"
    return "published_inferred"


def _extract_change_record_updated_at(
    soup: BeautifulSoup,
    field_values: dict[str, str],
    text_blob: str,
) -> datetime | None:
    candidates: list[str] = []
    for node in soup.find_all("time"):
        datetime_attr = node.get("datetime")
        if datetime_attr:
            candidates.append(str(datetime_attr))
        text = _normalize_ws(node.get_text(" ", strip=True))
        if text:
            candidates.append(text)

    for key, value in field_values.items():
        key_lower = key.lower()
        if any(token in key_lower for token in ("changed", "updated", "created", "date", "time")):
            candidates.append(value)

    for token in re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b", text_blob):
        candidates.append(token)

    parsed_values = [dt for dt in (_parse_datetime_value(item) for item in candidates) if dt is not None]
    if not parsed_values:
        return None
    return max(parsed_values)


def _extract_change_record_language(block) -> str:
    for node in [block, block.find("code") if hasattr(block, "find") else None]:
        if node is None:
            continue
        classes = node.get("class", [])
        if not isinstance(classes, list):
            continue
        for css_class in classes:
            css_class = str(css_class)
            if css_class.startswith("language-"):
                return css_class.split("-", 1)[1]
    return "php"


def _extract_change_record_code_blocks(container) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    if container is None:
        return blocks
    for pre in container.find_all("pre"):
        code_node = pre.find("code")
        code_text = (code_node or pre).get_text("\n")
        code_text = code_text.strip("\n")
        if len(code_text.strip()) < 20:
            continue

        label = ""
        for previous in pre.find_all_previous(["h2", "h3", "h4", "strong", "p"], limit=4):
            candidate = _normalize_ws(previous.get_text(" ", strip=True))
            if not candidate or len(candidate) > 140:
                continue
            label = candidate
            break

        blocks.append(
            {
                "label": label,
                "language": _extract_change_record_language(pre),
                "code": code_text,
            }
        )
    return blocks


def _extract_before_after_pairs(blocks: list[dict[str, str]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    pending_before: dict[str, str] | None = None

    for idx, block in enumerate(blocks):
        label = block.get("label", "")
        if BEFORE_HINT_RE.search(label):
            pending_before = block
            continue
        if AFTER_HINT_RE.search(label):
            if pending_before:
                pairs.append({"before": pending_before["code"], "after": block["code"], "language": block["language"]})
                pending_before = None
            elif idx > 0:
                pairs.append({"before": blocks[idx - 1]["code"], "after": block["code"], "language": block["language"]})

    if not pairs and len(blocks) >= 2:
        pairs.append({"before": blocks[0]["code"], "after": blocks[1]["code"], "language": blocks[1]["language"]})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        key = (pair.get("before", "").strip(), pair.get("after", "").strip())
        if not key[0] or not key[1] or key[0] == key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped


def _extract_change_record_rationale(container) -> str:
    if container is None:
        return ""
    paragraphs: list[str] = []
    for node in container.find_all(["p", "li"]):
        text = _normalize_ws(node.get_text(" ", strip=True))
        if len(text) < 60:
            continue
        lower = text.lower()
        if lower.startswith("before") or lower.startswith("after"):
            continue
        paragraphs.append(text)
        if len(paragraphs) >= 4:
            break
    return "\n\n".join(paragraphs)


def _render_change_record_markdown(
    *,
    title: str,
    record_url: str,
    status: str,
    versions: list[str],
    updated_at: datetime | None,
    rationale: str,
    pairs: list[dict[str, str]],
) -> str:
    timestamp = updated_at.isoformat().replace("+00:00", "Z") if updated_at else "unknown"
    lines: list[str] = [
        f"# {title}",
        "",
        "## Change Record Metadata",
        f"- Status: {status}",
        f"- Target versions: {', '.join(versions) if versions else 'unknown'}",
        f"- Updated: {timestamp}",
        f"- URL: {record_url}",
        "",
        "## Rationale",
        rationale or "No rationale text extracted from the record body.",
    ]

    for idx, pair in enumerate(pairs, start=1):
        language = pair.get("language", "php")
        before_heading = "## Before" if idx == 1 else f"## Before {idx}"
        after_heading = "## After" if idx == 1 else f"## After {idx}"
        lines.extend(
            [
                "",
                before_heading,
                f"```{language}",
                pair.get("before", "").rstrip(),
                "```",
                "",
                after_heading,
                f"```{language}",
                pair.get("after", "").rstrip(),
                "```",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _parse_change_record_page(
    html: str,
    record_url: str,
    *,
    target_versions: set[str],
    lookback_cutoff: datetime | None,
) -> dict[str, object]:
    soup = BeautifulSoup(html, "lxml")
    title_node = soup.find("h1")
    title = _normalize_ws(title_node.get_text(" ", strip=True) if title_node else "")
    if not title:
        title = f"Drupal change record {_extract_node_id(record_url)}"

    fields = _extract_change_record_fields(soup)
    text_blob = _normalize_ws(soup.get_text(" ", strip=True))
    status = _extract_change_record_status(fields, text_blob)
    if status == "draft":
        return {"accepted": False, "reason": "draft"}

    versions = _extract_change_record_versions(fields, text_blob)
    if target_versions and not set(versions).intersection(target_versions):
        return {"accepted": False, "reason": "version_filtered"}

    updated_at = _extract_change_record_updated_at(soup, fields, text_blob)
    if lookback_cutoff and updated_at and updated_at < lookback_cutoff:
        return {"accepted": False, "reason": "lookback_filtered"}
    if lookback_cutoff and updated_at is None:
        return {"accepted": False, "reason": "missing_updated_at"}

    container = (
        soup.select_one(".change-record-description")
        or soup.select_one(".field--name-body")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.body
    )
    rationale = _extract_change_record_rationale(container)
    blocks = _extract_change_record_code_blocks(container)
    pairs = _extract_before_after_pairs(blocks)
    if not pairs:
        return {"accepted": False, "reason": "missing_before_after"}

    markdown = _render_change_record_markdown(
        title=title,
        record_url=record_url,
        status=status,
        versions=versions,
        updated_at=updated_at,
        rationale=rationale,
        pairs=pairs,
    )
    return {
        "accepted": True,
        "reason": "accepted",
        "markdown": markdown,
        "status": status,
        "versions": versions,
        "updated_at": updated_at.isoformat().replace("+00:00", "Z") if updated_at else None,
        "title": title,
        "node_id": _extract_node_id(record_url),
    }


def fetch_drupal_change_records(source: dict, fetcher: DocumentationFetcher, docs_dir: Path, logger: PipelineLogger) -> dict:
    start_url = str(source.get("url", "https://www.drupal.org/list-changes/drupal")).strip()
    allowed_node_prefix = str(source.get("allowed_node_prefix", "https://www.drupal.org/node/")).strip()
    required_status = str(source.get("status", "published")).strip().lower()
    max_list_pages = max(1, int(source.get("max_list_pages", 80)))
    max_records = max(1, int(source.get("max_records", 1000)))
    record_workers = _clamp_workers(source.get("parallel_workers", 1), default=1, hard_max=12)
    record_request_delay_seconds = max(
        0.0,
        float(source.get("record_request_delay_seconds", fetcher.request_delay_seconds)),
    )
    record_request_timeout_seconds = max(
        1,
        int(source.get("record_request_timeout_seconds", fetcher.request_timeout_seconds)),
    )
    listing_request_delay_seconds = max(
        0.0,
        float(source.get("listing_request_delay_seconds", min(0.2, fetcher.request_delay_seconds))),
    )
    lookback_months = max(1, int(source.get("lookback_months", 24)))
    target_versions = {
        _canonicalize_change_record_version(str(version).strip())
        for version in source.get("target_versions", ["10.2.x", "10.3.x", "11.x"])
        if str(version).strip()
    }

    lookback_cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_months * 30)
    change_records_dir = docs_dir / "www_drupal_org" / "change_records"
    change_records_dir.mkdir(parents=True, exist_ok=True)

    discovered_urls: set[str] = set()
    failed_pages = 0
    consecutive_empty_pages = 0
    consecutive_failed_pages = 0

    for page_number in range(max_list_pages):
        page_url = start_url if page_number == 0 else f"{start_url}?page={page_number}"
        try:
            response = fetcher.session.get(page_url, timeout=record_request_timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            failed_pages += 1
            consecutive_failed_pages += 1
            logger.error(f"Failed to fetch change record listing page {page_url}: {exc}")
            if consecutive_failed_pages >= 3:
                break
            continue
        consecutive_failed_pages = 0

        soup = BeautifulSoup(response.text, "lxml")
        node_links: set[str] = set()
        for anchor in soup.select(".view-content a[href*='/node/'], main a[href*='/node/']"):
            href = anchor.get("href")
            if not href:
                continue
            full_url = urljoin(page_url, href).split("#")[0]
            if not full_url.startswith(allowed_node_prefix):
                continue
            if CHANGE_RECORD_NODE_RE.search(full_url):
                node_links.add(full_url)

        if not node_links:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= 2:
                break
        else:
            consecutive_empty_pages = 0
            discovered_urls.update(node_links)
            if len(discovered_urls) >= max_records * 3:
                break

        if listing_request_delay_seconds > 0:
            time.sleep(listing_request_delay_seconds)

    sorted_urls = sorted(discovered_urls, key=lambda value: int(_extract_node_id(value) or 0), reverse=True)
    candidate_urls = sorted_urls[: max(1, min(len(sorted_urls), max_records * 3))]
    written = 0
    bytes_written = 0
    parse_failures = 0
    kept_ids: set[str] = set()
    reason_counts: dict[str, int] = {}
    inferred_status_count = 0

    def _fetch_and_parse(record_url: str) -> dict[str, object]:
        try:
            response = fetcher.session.get(record_url, timeout=record_request_timeout_seconds)
            response.raise_for_status()
        except Exception as exc:
            return {"url": record_url, "accepted": False, "reason": "fetch_error", "error": str(exc), "parsed": {}}
        if record_request_delay_seconds > 0:
            time.sleep(record_request_delay_seconds)

        parsed = _parse_change_record_page(response.text, record_url, target_versions=target_versions, lookback_cutoff=lookback_cutoff)
        return {"url": record_url, "accepted": bool(parsed.get("accepted")), "reason": str(parsed.get("reason", "unknown")), "parsed": parsed}

    def _apply_parsed(payload: dict[str, object]) -> None:
        nonlocal parse_failures, written, bytes_written, inferred_status_count
        reason = str(payload.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason == "fetch_error":
            parse_failures += 1
            logger.error(f"Failed to fetch change record {payload.get('url')}: {payload.get('error')}")
            return

        parsed = payload.get("parsed", {})
        if not isinstance(parsed, dict):
            return
        if not bool(payload.get("accepted")):
            return

        parsed_status = str(parsed.get("status", "")).strip().lower()
        if required_status == "published" and parsed_status not in {"published", "published_inferred"}:
            reason_counts["status_filtered"] = reason_counts.get("status_filtered", 0) + 1
            return

        node_id = str(parsed.get("node_id") or "").strip()
        if not node_id or node_id == "unknown":
            node_id = _safe_filename_fragment(str(parsed.get("title", "")))
        filename = f"{node_id}.md"
        target_path = change_records_dir / filename
        markdown = str(parsed.get("markdown", ""))
        with open(target_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)

        kept_ids.add(target_path.stem)
        written += 1
        bytes_written += len(markdown.encode("utf-8"))
        if str(parsed.get("status", "")) == "published_inferred":
            inferred_status_count += 1

    if record_workers <= 1:
        for record_url in candidate_urls:
            if written >= max_records:
                break
            _apply_parsed(_fetch_and_parse(record_url))
    else:
        with ThreadPoolExecutor(max_workers=record_workers) as executor:
            for payload in executor.map(_fetch_and_parse, candidate_urls):
                if written >= max_records:
                    break
                _apply_parsed(payload)

    pruned_files = 0
    for existing in change_records_dir.glob("*.md"):
        if existing.stem not in kept_ids:
            existing.unlink(missing_ok=True)
            pruned_files += 1

    return {
        "success": written > 0,
        "pages": written,
        "bytes": bytes_written,
        "retried": 0,
        "failed_pages": failed_pages + parse_failures,
        "discovered_links": len(discovered_urls),
        "written_records": written,
        "pruned_records": pruned_files,
        "lookback_months": lookback_months,
        "target_versions": sorted(target_versions),
        "required_status": required_status,
        "reasons": reason_counts,
        "published_status_inferred": inferred_status_count,
        "record_workers": record_workers,
    }


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
    acq_cfg = config.get("acquisition", {})
    docs_cfg = acq_cfg.get("docs", {})
    parallel_cfg = acq_cfg.get("parallel", {})

    max_pages_per_source = int(docs_cfg.get("max_pages_per_source", 200))
    allowed_prefixes_cfg = docs_cfg.get("allowed_prefixes", {})
    url_denylist_terms = [str(term).strip() for term in docs_cfg.get("url_denylist_terms", []) if str(term).strip()]
    request_timeout_seconds = int(docs_cfg.get("request_timeout_seconds", 30))
    request_delay_seconds = float(docs_cfg.get("request_delay_seconds", 0.3))
    retry_backoff_seconds = float(docs_cfg.get("retry_backoff_seconds", 0.4))
    max_retries = int(docs_cfg.get("max_retries", 2))
    change_record_request_timeout_seconds = int(
        docs_cfg.get("change_record_request_timeout_seconds", request_timeout_seconds)
    )
    change_record_request_delay_seconds = float(
        docs_cfg.get("change_record_request_delay_seconds", request_delay_seconds)
    )
    change_record_listing_delay_seconds = float(
        docs_cfg.get("change_record_listing_delay_seconds", min(0.2, request_delay_seconds))
    )

    git_workers = _clamp_workers(parallel_cfg.get("git_workers", 4), default=4, hard_max=12)
    docs_workers = _clamp_workers(parallel_cfg.get("docs_workers", 2), default=2, hard_max=8)
    change_record_workers = _clamp_workers(parallel_cfg.get("change_record_workers", 4), default=4, hard_max=12)

    excluded_source_ids = {
        str(source_id).strip().lower() for source_id in acq_cfg.get("exclude_source_ids", []) if str(source_id).strip()
    }

    curated_sources = list(sources_data.get("sources", {}).get("curated", []))
    drupal_project_sources = list(sources_data.get("sources", {}).get("drupal_org_projects", []))

    def _run_git_source(source: dict, default_ref: str | None = None) -> dict:
        source_id = str(source.get("id"))
        url = str(source.get("url"))
        ref = source.get("ref", default_ref or "master")
        target_dir = repos_dir / source_id
        result = clone_or_fetch(url, ref, target_dir, logger)
        payload = {
            "source_id": source_id,
            "type": "git",
            "success": bool(result.get("success")),
            "status": "ok" if result.get("success") else "failed",
            "bytes": 0,
            "retried": int(result.get("retried", 0)),
            "action": result.get("action"),
            "commit": result.get("commit"),
            "pages": 0,
            "manifest_output_path": str(target_dir.relative_to(root)),
            "manifest_output_hash": str(result.get("commit")) if result.get("success") else None,
        }
        return payload

    def _build_fetcher() -> DocumentationFetcher:
        return DocumentationFetcher(
            logger,
            docs_dir,
            request_timeout_seconds=request_timeout_seconds,
            request_delay_seconds=request_delay_seconds,
            retry_backoff_seconds=retry_backoff_seconds,
            max_retries=max_retries,
        )

    def _run_doc_source(source: dict) -> dict:
        source_id = str(source.get("id"))
        source_type = str(source.get("type", "http"))
        url = str(source.get("url", ""))
        fetcher = _build_fetcher()
        try:
            if source_type == "http":
                allowed_prefixes = allowed_prefixes_cfg.get(source_id)
                if not isinstance(allowed_prefixes, list) or not allowed_prefixes:
                    allowed_prefixes = _default_prefix_for_url(url)

                logger.info(
                    f"Starting recursive fetch for {source_id}",
                    source_id=source_id,
                    allowed_prefixes=allowed_prefixes,
                    max_pages=max_pages_per_source,
                    url_denylist_terms=url_denylist_terms,
                )
                result = fetcher.recursive_fetch(
                    url,
                    allowed_prefixes=allowed_prefixes,
                    max_pages=max_pages_per_source,
                    url_denylist_terms=url_denylist_terms,
                )
                is_valid_doc_fetch = _doc_fetch_is_valid(result)
                return {
                    "source_id": source_id,
                    "type": "http",
                    "success": is_valid_doc_fetch,
                    "status": "ok"
                    if is_valid_doc_fetch
                    else "failed_zero_pages"
                    if int(result.get("pages", 0)) <= 0
                    else "failed",
                    "bytes": int(result.get("bytes", 0)),
                    "retried": int(result.get("retried", 0)),
                    "pages": int(result.get("pages", 0)),
                    "failed_pages": int(result.get("failed_pages", 0)),
                    "allowed_prefixes": allowed_prefixes,
                    "url_denylist_terms": url_denylist_terms,
                    "manifest_output_path": str((docs_dir / urlparse(url).netloc.replace(".", "_")).relative_to(root)),
                    "manifest_output_hash": "collection",
                }

            if source_type == "drupal_change_records":
                change_record_source = dict(source)
                change_record_source.setdefault("parallel_workers", change_record_workers)
                change_record_source.setdefault("record_request_timeout_seconds", change_record_request_timeout_seconds)
                change_record_source.setdefault("record_request_delay_seconds", change_record_request_delay_seconds)
                change_record_source.setdefault("listing_request_delay_seconds", change_record_listing_delay_seconds)

                logger.info(
                    f"Starting change record extraction for {source_id}",
                    source_id=source_id,
                    url=url,
                    max_list_pages=source.get("max_list_pages"),
                    max_records=source.get("max_records"),
                    worker_count=change_record_workers,
                )
                result = fetch_drupal_change_records(change_record_source, fetcher, docs_dir, logger)
                is_valid_doc_fetch = _doc_fetch_is_valid(result)
                return {
                    "source_id": source_id,
                    "type": "drupal_change_records",
                    "success": is_valid_doc_fetch,
                    "status": "ok"
                    if is_valid_doc_fetch
                    else "failed_zero_pages"
                    if int(result.get("pages", 0)) <= 0
                    else "failed",
                    "bytes": int(result.get("bytes", 0)),
                    "retried": int(result.get("retried", 0)),
                    "pages": int(result.get("pages", 0)),
                    "failed_pages": int(result.get("failed_pages", 0)),
                    "discovered_links": int(result.get("discovered_links", 0)),
                    "written_records": int(result.get("written_records", 0)),
                    "pruned_records": int(result.get("pruned_records", 0)),
                    "lookback_months": int(result.get("lookback_months", 0)),
                    "target_versions": result.get("target_versions", []),
                    "required_status": result.get("required_status", ""),
                    "reasons": result.get("reasons", {}),
                    "published_status_inferred": int(result.get("published_status_inferred", 0)),
                    "record_workers": int(result.get("record_workers", change_record_workers)),
                    "manifest_output_path": str((docs_dir / "www_drupal_org" / "change_records").relative_to(root)),
                    "manifest_output_hash": "collection",
                }

            return {
                "source_id": source_id,
                "type": source_type,
                "success": False,
                "status": "unsupported_source_type",
                "bytes": 0,
                "retried": 0,
                "pages": 0,
                "failed_pages": 0,
                "manifest_output_path": None,
                "manifest_output_hash": None,
            }
        finally:
            fetcher.session.close()

    git_job_inputs: list[tuple[dict, str | None]] = []
    doc_job_inputs: list[dict] = []

    for source in curated_sources:
        if str(source.get("id", "")).strip().lower() in excluded_source_ids:
            continue
        source_type = str(source.get("type", "")).strip()
        if source_type == "git":
            git_job_inputs.append((source, None))
        elif source_type in {"http", "drupal_change_records"}:
            doc_job_inputs.append(source)

    for source in drupal_project_sources:
        if str(source.get("id", "")).strip().lower() in excluded_source_ids:
            continue
        git_job_inputs.append((source, "master"))

    git_results: dict[str, dict] = {}
    if git_job_inputs:
        logger.info(
            "Running git acquisition tasks.",
            worker_count=git_workers,
            task_count=len(git_job_inputs),
        )
        with ThreadPoolExecutor(max_workers=git_workers) as executor:
            future_to_source_id = {
                executor.submit(_run_git_source, source, default_ref): str(source.get("id"))
                for source, default_ref in git_job_inputs
            }
            for future in as_completed(future_to_source_id):
                source_id = future_to_source_id[future]
                try:
                    git_results[source_id] = future.result()
                except Exception as exc:
                    logger.error("Git acquisition task failed.", source_id=source_id, error=str(exc))
                    git_results[source_id] = {
                        "source_id": source_id,
                        "type": "git",
                        "success": False,
                        "status": "failed_exception",
                        "bytes": 0,
                        "retried": 0,
                        "action": None,
                        "commit": None,
                        "pages": 0,
                        "manifest_output_path": str((repos_dir / source_id).relative_to(root)),
                        "manifest_output_hash": None,
                    }

    doc_results: dict[str, dict] = {}
    if doc_job_inputs:
        logger.info(
            "Running documentation acquisition tasks.",
            worker_count=docs_workers,
            task_count=len(doc_job_inputs),
        )
        if docs_workers <= 1:
            for source in doc_job_inputs:
                source_id = str(source.get("id"))
                try:
                    doc_results[source_id] = _run_doc_source(source)
                except Exception as exc:
                    logger.error("Documentation acquisition task failed.", source_id=source_id, error=str(exc))
                    doc_results[source_id] = {
                        "source_id": source_id,
                        "type": str(source.get("type", "http")),
                        "success": False,
                        "status": "failed_exception",
                        "bytes": 0,
                        "retried": 0,
                        "pages": 0,
                        "failed_pages": 1,
                        "manifest_output_path": None,
                        "manifest_output_hash": None,
                    }
        else:
            with ThreadPoolExecutor(max_workers=docs_workers) as executor:
                future_to_source_id = {executor.submit(_run_doc_source, source): str(source.get("id")) for source in doc_job_inputs}
                for future in as_completed(future_to_source_id):
                    source_id = future_to_source_id[future]
                    try:
                        doc_results[source_id] = future.result()
                    except Exception as exc:
                        logger.error("Documentation acquisition task failed.", source_id=source_id, error=str(exc))
                        doc_results[source_id] = {
                            "source_id": source_id,
                            "type": "http",
                            "success": False,
                            "status": "failed_exception",
                            "bytes": 0,
                            "retried": 0,
                            "pages": 0,
                            "failed_pages": 1,
                            "manifest_output_path": None,
                            "manifest_output_hash": None,
                        }

    success_count = 0
    failure_count = 0
    doc_pages_total = 0
    zero_page_doc_sources = 0
    fetch_status: list[dict] = []

    def _append_status(payload: dict) -> None:
        nonlocal success_count, failure_count, doc_pages_total, zero_page_doc_sources
        payload = dict(payload)
        source_type = str(payload.get("type", "unknown"))
        is_success = bool(payload.get("success"))
        status = str(payload.get("status", ""))
        pages = int(payload.get("pages", 0))

        if status == "skipped_excluded":
            pass
        elif is_success:
            success_count += 1
        else:
            failure_count += 1

        if source_type in {"http", "drupal_change_records"}:
            doc_pages_total += pages
            if status != "skipped_excluded" and not is_success and pages <= 0:
                zero_page_doc_sources += 1

        output_path = payload.pop("manifest_output_path", None)
        output_hash = payload.pop("manifest_output_hash", None)
        source_id = str(payload.get("source_id"))
        if is_success and output_path and output_hash:
            manifest.add_output(source_id, output_path, output_hash)
        fetch_status.append(payload)

    for source in curated_sources:
        source_id = str(source.get("id"))
        source_type = str(source.get("type", "unknown"))
        if source_id.strip().lower() in excluded_source_ids:
            _append_status(
                {
                    "source_id": source_id,
                    "type": source_type,
                    "success": True,
                    "status": "skipped_excluded",
                    "bytes": 0,
                    "retried": 0,
                    "pages": 0,
                    "manifest_output_path": None,
                    "manifest_output_hash": None,
                }
            )
            continue
        if source_type == "git":
            _append_status(git_results.get(source_id, {"source_id": source_id, "type": "git", "success": False, "status": "missing_result"}))
        elif source_type in {"http", "drupal_change_records"}:
            _append_status(
                doc_results.get(
                    source_id,
                    {"source_id": source_id, "type": source_type, "success": False, "status": "missing_result", "pages": 0},
                )
            )
        else:
            _append_status(
                {
                    "source_id": source_id,
                    "type": source_type,
                    "success": False,
                    "status": "unsupported_source_type",
                    "bytes": 0,
                    "retried": 0,
                    "pages": 0,
                    "manifest_output_path": None,
                    "manifest_output_hash": None,
                }
            )

    for source in drupal_project_sources:
        source_id = str(source.get("id"))
        source_type = str(source.get("type", "git"))
        if source_id.strip().lower() in excluded_source_ids:
            _append_status(
                {
                    "source_id": source_id,
                    "type": source_type,
                    "success": True,
                    "status": "skipped_excluded",
                    "bytes": 0,
                    "retried": 0,
                    "pages": 0,
                    "manifest_output_path": None,
                    "manifest_output_hash": None,
                }
            )
            continue
        _append_status(
            git_results.get(
                source_id,
                {
                    "source_id": source_id,
                    "type": "git",
                    "success": False,
                    "status": "missing_result",
                    "bytes": 0,
                    "retried": 0,
                    "pages": 0,
                    "manifest_output_path": str((repos_dir / source_id).relative_to(root)),
                    "manifest_output_hash": None,
                },
            )
        )

    manifest.data["fetch_status"] = fetch_status
    manifest.set_metrics(
        {
            "success_count": success_count,
            "failure_count": failure_count,
            "doc_pages_captured": doc_pages_total,
            "zero_page_doc_sources": zero_page_doc_sources,
            "git_workers": git_workers,
            "docs_workers": docs_workers,
            "change_record_workers": change_record_workers,
            "total_repos": len(drupal_project_sources)
            + sum(1 for source in curated_sources if source.get("type") == "git"),
        }
    )

    manifest.save()
    return 0 if failure_count == 0 else 1
