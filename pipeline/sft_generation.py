import json
import os
import re
from pathlib import Path

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash


DECLARATION_RE = re.compile(
    r"(?m)^[ \t]*(?:final\s+|abstract\s+)?(?:readonly\s+)?(class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
PHP_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*|#(?!\[)[^\n]*", re.DOTALL)
VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
SOURCE_FILE_PLACEHOLDER = "<source_file>"


class InstructionGenerator:
    def __init__(self, logger: PipelineLogger, config: dict | None = None):
        self.logger = logger
        self.config = config or {}
        self.enable_symbol_kind_prompts = bool(self.config.get("enable_symbol_kind_prompts", True))
        self.samples: list[dict] = []
        self._seen_pairs: set[tuple[str, str]] = set()

    def _append(self, sample: dict) -> None:
        key = (sample["instruction"], sample["output"])
        if key in self._seen_pairs:
            return
        self._seen_pairs.add(key)
        self.samples.append(sample)

    def _sanitize_symbol_name(self, candidate: str) -> str:
        parts = re.findall(r"[A-Za-z0-9]+", candidate)
        if not parts:
            return "UnknownSymbol"
        cleaned = "".join(part[:1].upper() + part[1:] for part in parts if part)
        if not cleaned[0].isalpha():
            cleaned = f"Symbol{cleaned}"
        return cleaned

    def _symbol_from_path(self, rel_path: str) -> str:
        stem = Path(rel_path).stem
        return self._sanitize_symbol_name(stem)

    def _kind_from_path(self, rel_path: str) -> str:
        base = Path(rel_path).name.lower()
        if base.endswith("interface.php"):
            return "interface"
        if base.endswith("trait.php"):
            return "trait"
        if base.endswith("enum.php"):
            return "enum"
        return "class"

    def _extract_symbol_from_php(self, content: str, rel_path: str) -> tuple[str, str, str]:
        sanitized = PHP_COMMENT_RE.sub("", content)
        matches = list(DECLARATION_RE.finditer(sanitized))
        if len(matches) == 1:
            declaration = matches[0]
            symbol_kind = declaration.group(1)
            symbol_name = declaration.group(2)
            if VALID_SYMBOL_RE.match(symbol_name):
                return symbol_kind, symbol_name, "declaration"

        # Multiple declarations can represent compound files; use deterministic
        # path fallback to avoid malformed instruction slots.
        return self._kind_from_path(rel_path), self._symbol_from_path(rel_path), "path_fallback"

    def generate_from_php(self, content: str, rel_path: str):
        symbol_kind, symbol_name, extraction_method = self._extract_symbol_from_php(content, rel_path)

        if self.enable_symbol_kind_prompts:
            instruction = (
                f"Show me the implementation of the {symbol_kind} {symbol_name} in the file {SOURCE_FILE_PLACEHOLDER}."
            )
        else:
            instruction = f"Show me the implementation of the class {symbol_name} in the file {SOURCE_FILE_PLACEHOLDER}."

        self._append(
            {
                "instruction": instruction,
                "input": "",
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "code_reference",
                    "symbol_kind": symbol_kind,
                    "symbol_name": symbol_name,
                    "symbol_extraction_method": extraction_method,
                },
            }
        )

    def generate_from_yaml(self, content: str, rel_path: str):
        stem = Path(rel_path).stem.replace("_", " ")
        instruction = f"Provide the Drupal 11 YAML configuration from {SOURCE_FILE_PLACEHOLDER}."
        self._append(
            {
                "instruction": instruction,
                "input": "",
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "yaml_reference",
                    "topic": stem,
                },
            }
        )

    def generate_from_twig(self, content: str, rel_path: str):
        instruction = f"Show the Twig template implementation in {SOURCE_FILE_PLACEHOLDER} for Drupal 11 theming."
        self._append(
            {
                "instruction": instruction,
                "input": "",
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "twig_reference",
                },
            }
        )

    def generate_from_doc(self, content: str, rel_path: str):
        generic_titles = {
            "contents of this file",
            "introduction",
            "readme",
            "license",
            "requirements",
            "installation",
            "configuration",
            "for developers",
            "description",
            "features",
            "support",
            "author",
            "maintainers",
            "copyright",
            "how it works",
            "prerequisites",
            "gnu general public license",
            "changelog",
            "release notes",
        }

        skip_patterns = [
            "license",
            "changelog",
            "copyright",
            "maintainers",
            "fixtures",
            "node_modules",
            "vendor/",
        ]
        rel_lower = rel_path.lower()
        if any(pattern in rel_lower for pattern in skip_patterns):
            return

        title = ""
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        if not title:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            title = lines[0].strip("# ") if lines else Path(rel_path).stem

        title = re.sub(r"\{#.*?\}", "", title).strip()
        title_lower = title.lower()

        if len(title) < 5 or title_lower in generic_titles:
            return
        if any(token in title_lower for token in ["cookie", "sign in", "tracking", "web beacon"]):
            return
        if "drupal 7" in content.lower() and "drupal 11" not in content.lower() and "drupal 10" not in content.lower():
            return

        instruction = f"Explain the following topic based on Drupal 11 documentation: {title}"
        self._append(
            {
                "instruction": instruction,
                "input": "",
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "doc_summary",
                    "topic": title,
                },
            }
        )

    def save(self, output_path: Path):
        with open(output_path, "w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")


def run_sft_generation_stage(config: dict, logger: PipelineLogger, root: Path):
    clean_dir = root / "clean"
    sft_dir = root / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest("sft_generation", sft_dir)

    sft_cfg = config.get("sft_generation", {})
    include_extensions = sft_cfg.get(
        "include_extensions",
        [".php", ".module", ".inc", ".install", ".theme", ".yml", ".twig", ".md"],
    )
    include_extensions = {suffix.lower() for suffix in include_extensions}

    generator = InstructionGenerator(logger, config=sft_cfg)

    for dirpath, _, filenames in os.walk(clean_dir):
        for filename in filenames:
            if filename == "dedup_manifest.json":
                continue

            clean_path = Path(dirpath) / filename
            if clean_path.suffix.lower() not in include_extensions:
                continue

            rel_path = str(clean_path.relative_to(clean_dir))
            try:
                with open(clean_path, "r", encoding="utf-8") as handle:
                    content = handle.read()

                suffix = clean_path.suffix.lower()
                if suffix in {".php", ".module", ".inc", ".install", ".theme"}:
                    generator.generate_from_php(content, rel_path)
                elif suffix == ".yml":
                    generator.generate_from_yaml(content, rel_path)
                elif suffix == ".twig":
                    generator.generate_from_twig(content, rel_path)
                elif suffix in {".md", ".txt", ".html"}:
                    generator.generate_from_doc(content, rel_path)
            except Exception as exc:
                logger.error(f"Error generating SFT samples for {clean_path}: {str(exc)}")

    output_file = sft_dir / "combined.jsonl"
    generator.save(output_file)

    manifest.set_metrics({"total_samples": len(generator.samples)})
    manifest.add_output("combined_sft", "sft/combined.jsonl", calculate_hash(output_file))
    manifest.save()

    logger.info(f"SFT generation complete. Generated {len(generator.samples)} samples.")
    return 0
