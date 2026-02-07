import hashlib
import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash


class Normalizer:
    def __init__(self, logger: PipelineLogger, dedup_cfg: dict | None = None):
        self.logger = logger
        self.dedup_cfg = dedup_cfg or {}

        self.exact_hash_enabled = bool(self.dedup_cfg.get("exact_hash", True))
        self.near_dup_enabled = bool(self.dedup_cfg.get("near_dup_enabled", False))
        self.near_dup_method = str(self.dedup_cfg.get("near_dup_method", "simhash_5gram"))
        self.near_dup_threshold = float(self.dedup_cfg.get("near_dup_threshold", 0.92))

        self.seen_hashes: dict[str, tuple[str, str]] = {}
        self.family_fingerprints: dict[str, list[tuple[int, str]]] = {}
        self.family_lsh: dict[str, dict[tuple[int, int], list[int]]] = {}
        self.dedup_entries: list[dict] = []

        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "deduplicated_files": 0,
            "near_deduplicated_files": 0,
            "rejected_files": 0,
            "bytes_saved": 0,
        }

    def normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        normalized = "\n".join(lines).strip()
        return normalized + "\n"

    def strip_php_license(self, content: str) -> str:
        patterns = [r"/\*\*.*?@license.*?GPL.*? \*/"]
        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        return content

    def clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "lxml")

        noise_selectors = [
            "nav",
            "footer",
            "header",
            "aside",
            "script",
            "style",
            "noscript",
            ".region-header",
            ".region-footer",
            ".region-sidebar-first",
            ".region-sidebar-second",
            ".breadcrumb",
            ".visually-hidden",
            "#skip-link",
            ".cookie-banner",
            ".eu-cookie-compliance-banner",
            ".messages--warning",
            ".messages--error",
            ".search-block-form",
            ".navigation",
            ".contextual",
            ".social-media-links",
            ".feedback-link",
            "#drupal-live-announce",
            ".field--name-uid",
            ".field--name-created",
            "#block-api-drupal-org-cookieconsent",
            ".api-nav-tabs",
            "#block-bluecheese-branding",
        ]
        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        for element in soup.find_all(string=re.compile(r"Can we use first and third party cookies")):
            parent = element.find_parent()
            if parent:
                parent.decompose()

        content_selectors = [
            "article",
            ".main-content",
            ".documentation-content",
            "#main-content",
            ".node__content",
            ".field--name-body",
            ".api-content",
            "#block-system-main",
        ]

        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body if soup.body else soup

        h1 = soup.find("h1")
        if h1 and not main_content.find("h1"):
            main_content.insert(0, h1)

        markdown = md(str(main_content), heading_style="ATX")
        markdown = re.sub(r"\n\s*\n", "\n\n", markdown)
        markdown = re.sub(r"\[Edit\].*?\n", "", markdown)
        return markdown.strip()

    def _content_family(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".php", ".module", ".inc", ".install", ".profile", ".theme"}:
            return "php"
        if suffix in {".md", ".txt", ".html"}:
            return "doc"
        if suffix == ".yml":
            return "yaml"
        if suffix == ".twig":
            return "twig"
        return "other"

    def _simhash_5gram(self, text: str) -> int:
        compact = re.sub(r"\s+", " ", text.lower()).strip()
        if len(compact) < 5:
            compact = (compact + " " * 5)[:5]
        shingles = [compact[i : i + 5] for i in range(max(1, len(compact) - 4))]

        bits = [0] * 64
        for shingle in shingles:
            digest = hashlib.sha256(shingle.encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big")
            for bit in range(64):
                mask = 1 << bit
                bits[bit] += 1 if (value & mask) else -1

        fingerprint = 0
        for bit, score in enumerate(bits):
            if score >= 0:
                fingerprint |= 1 << bit
        return fingerprint

    def _hamming_similarity(self, left: int, right: int) -> float:
        xor = left ^ right
        distance = xor.bit_count()
        return 1.0 - (distance / 64.0)

    def _near_duplicate_of(self, family: str, fingerprint: int) -> tuple[str | None, float]:
        lsh = self.family_lsh.setdefault(family, {})
        fingerprints = self.family_fingerprints.setdefault(family, [])

        candidate_indices: set[int] = set()
        for block in range(4):
            shift = block * 16
            key = (block, (fingerprint >> shift) & 0xFFFF)
            for idx in lsh.get(key, []):
                candidate_indices.add(idx)

        best_path = None
        best_similarity = 0.0
        for idx in candidate_indices:
            candidate_fp, candidate_path = fingerprints[idx]
            similarity = self._hamming_similarity(fingerprint, candidate_fp)
            if similarity > best_similarity:
                best_similarity = similarity
                best_path = candidate_path

        return best_path, best_similarity

    def _register_fingerprint(self, family: str, fingerprint: int, canonical_path: str) -> None:
        lsh = self.family_lsh.setdefault(family, {})
        fingerprints = self.family_fingerprints.setdefault(family, [])
        idx = len(fingerprints)
        fingerprints.append((fingerprint, canonical_path))
        for block in range(4):
            shift = block * 16
            key = (block, (fingerprint >> shift) & 0xFFFF)
            lsh.setdefault(key, []).append(idx)

    def process_file(self, raw_path: Path, clean_dir: Path, root: Path) -> bool:
        self.stats["total_files"] += 1

        if raw_path.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
        ]:
            return False

        try:
            with open(raw_path, "rb") as handle:
                raw_bytes = handle.read()

            if b"\x00" in raw_bytes and raw_path.suffix.lower() not in [".php", ".module", ".inc"]:
                return False

            content = raw_bytes.decode("utf-8", errors="ignore")
            original_size = len(raw_bytes)

            if raw_path.suffix.lower() == ".html":
                content = self.clean_html(content)
                if len(content) < 200:
                    self.stats["rejected_files"] += 1
                    return False
                doc_lower = content.lower()
                if "drupal 7" in doc_lower and "drupal 11" not in doc_lower and "drupal 10" not in doc_lower:
                    if "benchmarking and profiling drupal" not in doc_lower:
                        self.stats["rejected_files"] += 1
                        return False

            elif raw_path.suffix.lower() in [".php", ".module", ".inc", ".install", ".profile", ".theme"]:
                content = self.strip_php_license(content)

            content = self.normalize_text(content)
            family = self._content_family(raw_path)

            rel_path = raw_path.relative_to(root / "raw")
            rel_path_str = str(rel_path)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            if self.exact_hash_enabled and content_hash in self.seen_hashes:
                canonical_source, canonical_family = self.seen_hashes[content_hash]
                self.stats["deduplicated_files"] += 1
                self.stats["bytes_saved"] += original_size
                self.dedup_entries.append(
                    {
                        "path": rel_path_str,
                        "canonical_source": canonical_source,
                        "family": canonical_family,
                        "dup_type": "exact",
                        "similarity_score": 1.0,
                        "hash": content_hash,
                    }
                )
                return False

            if self.near_dup_enabled and self.near_dup_method == "simhash_5gram":
                fingerprint = self._simhash_5gram(content)
                near_path, similarity = self._near_duplicate_of(family, fingerprint)
                if near_path and similarity >= self.near_dup_threshold:
                    self.stats["near_deduplicated_files"] += 1
                    self.stats["bytes_saved"] += original_size
                    self.dedup_entries.append(
                        {
                            "path": rel_path_str,
                            "canonical_source": near_path,
                            "family": family,
                            "dup_type": "near",
                            "similarity_score": round(similarity, 4),
                            "hash": content_hash,
                        }
                    )
                    return False
                self._register_fingerprint(family, fingerprint, rel_path_str)

            target_path = clean_dir / rel_path
            if target_path.suffix.lower() == ".html":
                target_path = target_path.with_suffix(".md")
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write(content)

            self.seen_hashes[content_hash] = (rel_path_str, family)
            self.dedup_entries.append(
                {
                    "path": rel_path_str,
                    "canonical_source": rel_path_str,
                    "family": family,
                    "dup_type": "canonical",
                    "similarity_score": 1.0,
                    "hash": content_hash,
                }
            )
            self.stats["processed_files"] += 1
            return True

        except Exception as exc:
            self.logger.error(f"Error processing {raw_path}: {str(exc)}")
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

    dedup_cfg = config.get("normalization", {}).get("dedup", {})
    normalizer = Normalizer(logger, dedup_cfg=dedup_cfg)

    raw_root = root / "raw"
    for dirpath, _, filenames in os.walk(raw_root):
        for filename in filenames:
            raw_path = Path(dirpath) / filename
            normalizer.process_file(raw_path, clean_dir, root)

    manifest.set_metrics(normalizer.stats)

    dedup_manifest_path = clean_dir / "dedup_manifest.json"
    dedup_payload = {
        "entries": normalizer.dedup_entries,
        "summary": {
            "total_entries": len(normalizer.dedup_entries),
            "canonical_count": sum(1 for entry in normalizer.dedup_entries if entry["dup_type"] == "canonical"),
            "exact_dup_count": sum(1 for entry in normalizer.dedup_entries if entry["dup_type"] == "exact"),
            "near_dup_count": sum(1 for entry in normalizer.dedup_entries if entry["dup_type"] == "near"),
        },
    }
    with open(dedup_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(dedup_payload, handle, indent=2)

    manifest.add_output("dedup_manifest", "clean/dedup_manifest.json", calculate_hash(dedup_manifest_path))
    manifest.save()

    logger.info(f"Normalization complete. Metrics: {normalizer.stats}")
    return 0
