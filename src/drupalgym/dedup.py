from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir, list_files, sha256_text, write_json


@dataclass(frozen=True)
class DedupConfig:
    clean_root: Path
    dedup_root: Path
    manifest_path: Path
    ignore_paths: tuple[Path, ...] = ()


def deduplicate_sources(config: DedupConfig) -> list[Path]:
    ensure_dir(config.dedup_root)
    seen: dict[str, str] = {}
    output: list[Path] = []
    records: list[dict[str, str]] = []

    for src_path in list_files(config.clean_root):
        if src_path in config.ignore_paths:
            continue
        rel_path = src_path.relative_to(config.clean_root)
        content = src_path.read_text(encoding="utf-8", errors="ignore")
        digest = sha256_text(content)
        if digest in seen:
            records.append(
                {
                    "source": str(rel_path),
                    "duplicate_of": seen[digest],
                    "hash": digest,
                }
            )
            continue
        dest_path = config.dedup_root / rel_path
        ensure_dir(dest_path.parent)
        dest_path.write_text(content, encoding="utf-8")
        seen[digest] = str(rel_path)
        records.append(
            {
                "source": str(rel_path),
                "deduped": str(rel_path),
                "hash": digest,
            }
        )
        output.append(dest_path)

    write_json(config.manifest_path, {"records": records})
    return output
