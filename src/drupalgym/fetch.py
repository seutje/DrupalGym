from __future__ import annotations

import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .utils import ManifestEntry, ensure_dir, list_files, write_json


@dataclass(frozen=True)
class AcquisitionConfig:
    raw_root: Path
    manifest_path: Path
    throttle_seconds: float = 0.5


def acquire_sources(entries: list[ManifestEntry], config: AcquisitionConfig) -> None:
    ensure_dir(config.raw_root)
    raw_manifest: list[dict[str, str | None]] = []
    for entry in entries:
        if entry.source == "git" and entry.url:
            repo_path = config.raw_root / "code" / entry.name
            commit = _clone_or_fetch(entry.url, repo_path)
            raw_manifest.append(
                {
                    "name": entry.name,
                    "source": entry.source,
                    "url": entry.url,
                    "commit": commit,
                }
            )
        elif entry.source == "doc" and entry.url:
            doc_path = _download_doc(entry.url, config.raw_root / "docs")
            raw_manifest.append(
                {
                    "name": entry.name,
                    "source": entry.source,
                    "url": entry.url,
                    "path": str(doc_path.relative_to(config.raw_root)),
                }
            )
        time.sleep(config.throttle_seconds)
    write_json(config.manifest_path, {"entries": raw_manifest})


def _clone_or_fetch(repo_url: str, repo_path: Path) -> str | None:
    ensure_dir(repo_path.parent)
    if repo_path.exists():
        _run(["git", "-C", str(repo_path), "fetch", "--depth", "1", "origin"])
    else:
        _run(["git", "clone", "--depth", "1", repo_url, str(repo_path)])
    result = _run(["git", "-C", str(repo_path), "rev-parse", "HEAD"], capture=True)
    return result.strip() if result else None


def _download_doc(url: str, docs_root: Path) -> Path:
    ensure_dir(docs_root)
    parts = urllib.parse.urlparse(url)
    path = parts.path.strip("/") or "index"
    target = docs_root / parts.netloc / f"{path}.html"
    ensure_dir(target.parent)
    request = urllib.request.Request(url, headers={"User-Agent": "DrupalGym/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        target.write_bytes(response.read())
    return target


def _run(cmd: list[str], capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture,
    )
    if capture:
        return result.stdout
    return ""


def list_raw_files(raw_root: Path) -> list[Path]:
    return list(list_files(raw_root))
