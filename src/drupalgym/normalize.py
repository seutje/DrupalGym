from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir, list_files, write_json


@dataclass(frozen=True)
class NormalizeConfig:
    raw_root: Path
    clean_root: Path
    manifest_path: Path


LICENSE_PATTERN = re.compile(r"^\s*/\*.*?\*/\s*", re.DOTALL)


def normalize_sources(config: NormalizeConfig) -> list[Path]:
    ensure_dir(config.clean_root)
    normalized: list[dict[str, str]] = []
    written: list[Path] = []

    for src_path in list_files(config.raw_root):
        rel_path = src_path.relative_to(config.raw_root)
        dest_path = config.clean_root / rel_path
        ensure_dir(dest_path.parent)
        content = src_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = _normalize_text(content, src_path.suffix)
        dest_path.write_text(cleaned, encoding="utf-8")
        normalized.append(
            {
                "source": str(rel_path),
                "normalized": str(dest_path.relative_to(config.clean_root)),
            }
        )
        written.append(dest_path)

    write_json(config.manifest_path, {"normalized": normalized})
    return written


def _normalize_text(text: str, suffix: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if suffix in {".php", ".module", ".inc", ".install"}:
        cleaned = _strip_license_header(cleaned)
        cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    if suffix in {".html", ".md", ".txt"}:
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip() + "\n"


def _strip_license_header(text: str) -> str:
    match = LICENSE_PATTERN.match(text)
    if match and "license" in match.group(0).lower():
        return text[match.end() :]
    return text
