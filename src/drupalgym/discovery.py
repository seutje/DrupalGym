from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import ManifestEntry, write_json


@dataclass(frozen=True)
class DiscoveryConfig:
    output_manifest: Path
    drupal_core_repo: str
    project_api_urls: list[str]
    doc_urls: list[str]
    core_constraint: str = "^11"
    throttle_seconds: float = 0.5


def _http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "DrupalGym/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _constraint_matches(constraint: str, required: str) -> bool:
    normalized = constraint.strip().lower()
    required_version = required.strip("^").lower()
    if required_version in normalized:
        return True
    return bool(re.search(rf"\b{re.escape(required_version)}(\.|\b)", normalized))


def _parse_project_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": entry.get("title") or entry.get("name") or "unknown",
        "composer": entry.get("field_project_machine_name") or entry.get("composer"),
        "core_constraint": entry.get("field_core_requirement")
        or entry.get("core_requirement"),
        "repo_url": entry.get("field_git_repo") or entry.get("repo"),
        "nid": entry.get("nid"),
    }


def discover_sources(config: DiscoveryConfig) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = [
        ManifestEntry(name="drupal-core", source="git", url=config.drupal_core_repo)
    ]

    for api_url in config.project_api_urls:
        payload = _http_get(api_url)
        data = _safe_json(payload)
        items = data.get("list") or data.get("projects") or []
        for item in items:
            parsed = _parse_project_entry(item)
            constraint = parsed.get("core_constraint") or ""
            if constraint and not _constraint_matches(constraint, config.core_constraint):
                continue
            repo_url = parsed.get("repo_url")
            if not repo_url:
                continue
            name = parsed.get("composer") or parsed.get("title")
            entries.append(ManifestEntry(name=name, source="git", url=repo_url))
        time.sleep(config.throttle_seconds)

    for doc_url in config.doc_urls:
        entries.append(ManifestEntry(name=_slugify(doc_url), source="doc", url=doc_url))

    write_json(
        config.output_manifest,
        {
            "sources": [entry.to_dict() for entry in entries],
            "core_constraint": config.core_constraint,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return entries


def _safe_json(payload: str) -> dict[str, Any]:
    try:
        return __import__("json").loads(payload)
    except Exception:
        return {}


def _slugify(url: str) -> str:
    parts = urllib.parse.urlparse(url)
    slug = f"{parts.netloc}{parts.path}".strip("/")
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", slug)
