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
NAMESPACE_RE = re.compile(r"(?m)^namespace\s+([^;]+);")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
DOC_MENU_IMAGE_RE = re.compile(r"^\[!\[[^\]]+\]\([^)]+\)\]\([^)]+\)\s*$")
DOC_BRANCH_LINE_RE = re.compile(r"^\d+\.\s+\[[^\]]+\]\([^)]+\)\s+\\\w+")


class InstructionGenerator:
    def __init__(self, logger: PipelineLogger, config: dict | None = None):
        self.logger = logger
        self.config = config or {}
        self.enable_symbol_kind_prompts = bool(self.config.get("enable_symbol_kind_prompts", True))
        self.doc_source_allowlist_prefixes = [
            str(prefix).strip().lower()
            for prefix in self.config.get(
                "doc_source_allowlist_prefixes",
                ["docs/www_drupal_org/", "docs/api_drupal_org/", "repos/drupal_core/"],
            )
            if str(prefix).strip()
        ]
        self.doc_topic_denylist_terms = [
            str(term).strip().lower()
            for term in self.config.get(
                "doc_topic_denylist_terms",
                ["mcp", "apidog", "gitlab duo", "workflow automation"],
            )
            if str(term).strip()
        ]
        self.doc_max_output_chars = int(self.config.get("doc_max_output_chars", 6000))
        self.sdc_bundle_variants = max(1, int(self.config.get("sdc_bundle_variants", 4)))
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

    @staticmethod
    def _file_hint(rel_path: str) -> str:
        return Path(rel_path).name or "unknown.file"

    @staticmethod
    def _module_hint(rel_path: str) -> str:
        parts = [part for part in Path(rel_path).parts if part]
        if len(parts) >= 2 and parts[0] == "repos":
            return parts[1]
        if len(parts) >= 2 and parts[0] == "docs":
            return parts[1]
        return parts[0] if parts else "unknown"

    @staticmethod
    def _extract_namespace(content: str) -> str:
        namespace_match = NAMESPACE_RE.search(content)
        if namespace_match:
            return namespace_match.group(1).strip()
        return "Drupal\\Custom"

    @staticmethod
    def _strip_doc_noise(content: str) -> str:
        lines: list[str] = []
        skip_next_blank = False
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if DOC_MENU_IMAGE_RE.match(stripped):
                continue
            if stripped.lower() == "same name and namespace in other branches":
                skip_next_blank = True
                continue
            if DOC_BRANCH_LINE_RE.match(stripped):
                continue
            if skip_next_blank and not stripped:
                skip_next_blank = False
                continue
            if stripped.lower().startswith("main menu"):
                continue
            lines.append(line)

        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _doc_paragraphs(content: str) -> list[str]:
        paragraphs: list[str] = []
        for block in re.split(r"\n\s*\n", content):
            candidate = block.strip()
            if not candidate:
                continue
            if len(candidate) < 50:
                continue
            if candidate.startswith("#"):
                continue
            if candidate.count("http") >= 2:
                continue
            if candidate.startswith("|"):
                continue
            candidate = MARKDOWN_LINK_RE.sub(r"\1", candidate)
            paragraphs.append(candidate)
        return paragraphs

    def _build_doc_output(self, content: str, title: str) -> str:
        cleaned = self._strip_doc_noise(content)
        paragraphs = self._doc_paragraphs(cleaned)
        selected = paragraphs[:6]
        if not selected:
            fallback = cleaned[: self.doc_max_output_chars].strip()
            return fallback

        chunks: list[str] = [f"Topic: {title}"]
        chunks.extend(selected)
        output = "\n\n".join(chunks)
        return output[: self.doc_max_output_chars].strip()

    def _is_doc_source_allowed(self, rel_path: str) -> bool:
        if not self.doc_source_allowlist_prefixes:
            return True
        rel_lower = rel_path.lower()
        return any(rel_lower.startswith(prefix) for prefix in self.doc_source_allowlist_prefixes)

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

    @staticmethod
    def _sdc_component_name_from_path(rel_path: str) -> str:
        path = Path(rel_path)
        stem = path.stem
        if stem.endswith(".component"):
            stem = stem[: -len(".component")]
        if stem and stem.lower() != "component":
            return stem
        return path.parent.name or "component"

    @staticmethod
    def _sdc_twig_candidates_for_yaml(rel_path: str) -> list[str]:
        path = Path(rel_path)
        candidates: list[str] = []
        if path.name.endswith(".component.yml"):
            candidates.append(str(path.with_suffix("").with_suffix(".twig")))
        if path.name == "component.yml":
            candidates.append(str(path.with_name("component.twig")))
            if path.parent.name:
                candidates.append(str(path.with_name(f"{path.parent.name}.twig")))
        return candidates

    @staticmethod
    def _sdc_yaml_candidates_for_twig(rel_path: str) -> list[str]:
        path = Path(rel_path)
        candidates: list[str] = []
        if path.suffix == ".twig":
            candidates.append(str(path.with_suffix(".component.yml")))
            candidates.append(str(path.with_name("component.yml")))
        return candidates

    def generate_from_php(self, content: str, rel_path: str):
        symbol_kind, symbol_name, extraction_method = self._extract_symbol_from_php(content, rel_path)
        namespace_value = self._extract_namespace(content)
        source_file = self._file_hint(rel_path)
        module_hint = self._module_hint(rel_path)

        if self.enable_symbol_kind_prompts:
            instruction = (
                f"Show me the implementation of the {symbol_kind} {symbol_name} in the file {SOURCE_FILE_PLACEHOLDER}."
            )
        else:
            instruction = f"Show me the implementation of the class {symbol_name} in the file {SOURCE_FILE_PLACEHOLDER}."

        input_text = "\n".join(
            [
                f"Source file: {SOURCE_FILE_PLACEHOLDER}",
                f"File name hint: {source_file}",
                f"Module hint: {module_hint}",
                f"Target symbol: {symbol_kind} {symbol_name}",
                f"Namespace: {namespace_value}",
            ]
        )
        self._append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "code_reference",
                    "sample_type": "retrieval",
                    "symbol_kind": symbol_kind,
                    "symbol_name": symbol_name,
                    "symbol_extraction_method": extraction_method,
                },
            }
        )

    def generate_from_yaml(self, content: str, rel_path: str):
        stem = Path(rel_path).stem.replace("_", " ")
        source_file = self._file_hint(rel_path)
        module_hint = self._module_hint(rel_path)
        instruction = f"Provide the Drupal 11 YAML configuration from {SOURCE_FILE_PLACEHOLDER}."
        input_text = "\n".join(
            [
                f"Source file: {SOURCE_FILE_PLACEHOLDER}",
                f"File name hint: {source_file}",
                f"Module hint: {module_hint}",
                f"Configuration topic: {stem}",
                "Task: Return the complete Drupal 11 YAML configuration.",
            ]
        )
        self._append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "yaml_reference",
                    "sample_type": "retrieval",
                    "topic": stem,
                },
            }
        )

    def generate_from_twig(self, content: str, rel_path: str):
        instruction = f"Show the Twig template implementation in {SOURCE_FILE_PLACEHOLDER} for Drupal 11 theming."
        source_file = self._file_hint(rel_path)
        module_hint = self._module_hint(rel_path)
        input_text = "\n".join(
            [
                f"Source file: {SOURCE_FILE_PLACEHOLDER}",
                f"File name hint: {source_file}",
                f"Module hint: {module_hint}",
                "Task: Return the complete Drupal 11 Twig template implementation.",
            ]
        )
        self._append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": content,
                "metadata": {
                    "source": rel_path,
                    "type": "twig_reference",
                    "sample_type": "retrieval",
                },
            }
        )

    def generate_sdc_bundle(
        self,
        *,
        yaml_rel_path: str,
        yaml_content: str,
        twig_rel_path: str,
        twig_content: str,
    ) -> None:
        component_name = self._sdc_component_name_from_path(yaml_rel_path)
        source_file = self._file_hint(yaml_rel_path)
        module_hint = self._module_hint(yaml_rel_path)
        instruction = (
            f"Show a Drupal 11 Single Directory Component implementation from {SOURCE_FILE_PLACEHOLDER}."
        )
        base_input = "\n".join(
            [
                f"Source file: {SOURCE_FILE_PLACEHOLDER}",
                f"File name hint: {source_file}",
                f"Module hint: {module_hint}",
                f"Component name: {component_name}",
                "Requirements:",
                "- Include both component.yml and Twig template content.",
                "- Keep the output aligned with Drupal 11 Single Directory Component conventions.",
            ]
        )
        output_bundle = "\n".join(
            [
                f"# {Path(yaml_rel_path).name}",
                yaml_content.rstrip(),
                "",
                f"# {Path(twig_rel_path).name}",
                twig_content.rstrip(),
                "",
            ]
        )
        variants: list[tuple[str, str, str]] = []
        variants.append((instruction, f"{base_input}\nVariant: files_bundle", output_bundle))

        output_with_tree = "\n".join(
            [
                "Component directory:",
                f"- {Path(yaml_rel_path).name}",
                f"- {Path(twig_rel_path).name}",
                "",
                output_bundle.rstrip(),
                "",
            ]
        )
        variants.append(
            (
                "Provide a Drupal 11 Single Directory Component reference including file structure and implementation.",
                f"{base_input}\nVariant: structure_and_files",
                output_with_tree,
            )
        )

        output_with_usage = "\n".join(
            [
                output_bundle.rstrip(),
                "# Example usage",
                "{# In a parent template #}",
                f"{{% include '@components/{component_name}/{Path(twig_rel_path).name}' with {{",
                "  title: 'Example',",
                "} %}}",
                "",
            ]
        )
        variants.append(
            (
                "Show a Drupal 11 SDC example with component.yml, Twig template, and usage snippet.",
                f"{base_input}\nVariant: files_and_usage",
                output_with_usage,
            )
        )

        output_with_contract = "\n".join(
            [
                output_bundle.rstrip(),
                "# Component contract checklist",
                "- component.yml defines props and slots.",
                "- Twig template references component props safely.",
                "- Structure follows Drupal 11 Single Directory Component rules.",
                "",
            ]
        )
        variants.append(
            (
                "Document a Drupal 11 SDC contract with component.yml and Twig implementation details.",
                f"{base_input}\nVariant: files_and_contract",
                output_with_contract,
            )
        )

        for variant_instruction, variant_input, variant_output in variants[: self.sdc_bundle_variants]:
            self._append(
                {
                    "instruction": variant_instruction,
                    "input": variant_input,
                    "output": variant_output,
                    "metadata": {
                        "source": yaml_rel_path,
                        "type": "sdc_reference",
                        "sample_type": "retrieval",
                        "topic": component_name,
                        "related_template": twig_rel_path,
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
        if not self._is_doc_source_allowed(rel_path):
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
        if any(term in title_lower for term in self.doc_topic_denylist_terms):
            return
        if any(token in title_lower for token in ["cookie", "sign in", "tracking", "web beacon"]):
            return
        if "drupal 7" in content.lower() and "drupal 11" not in content.lower() and "drupal 10" not in content.lower():
            return

        output_text = self._build_doc_output(content, title)
        if len(output_text) < 120:
            return

        instruction = f"Explain the following topic based on Drupal 11 documentation: {title}"
        source_file = self._file_hint(rel_path)
        module_hint = self._module_hint(rel_path)
        input_text = "\n".join(
            [
                f"Source file: {SOURCE_FILE_PLACEHOLDER}",
                f"File name hint: {source_file}",
                f"Domain hint: {module_hint}",
                "Task: Explain this Drupal 11 topic clearly and concisely.",
            ]
        )
        self._append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "metadata": {
                    "source": rel_path,
                    "type": "doc_summary",
                    "sample_type": "retrieval",
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
    enable_sdc_bundle_generation = bool(sft_cfg.get("enable_sdc_bundle_generation", True))

    generator = InstructionGenerator(logger, config=sft_cfg)
    content_by_rel_path: dict[str, str] = {}
    discovered_paths: list[str] = []

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
                content_by_rel_path[rel_path] = content
                discovered_paths.append(rel_path)

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

    if enable_sdc_bundle_generation:
        for rel_path in discovered_paths:
            rel_path_lower = rel_path.lower()
            if not rel_path_lower.endswith((".yml", ".twig")):
                continue

            yaml_rel: str | None = None
            twig_rel: str | None = None
            if rel_path_lower.endswith(".yml"):
                for candidate in generator._sdc_twig_candidates_for_yaml(rel_path):
                    if candidate in content_by_rel_path:
                        yaml_rel = rel_path
                        twig_rel = candidate
                        break
            elif rel_path_lower.endswith(".twig"):
                for candidate in generator._sdc_yaml_candidates_for_twig(rel_path):
                    if candidate in content_by_rel_path:
                        yaml_rel = candidate
                        twig_rel = rel_path
                        break

            if not yaml_rel or not twig_rel:
                continue
            generator.generate_sdc_bundle(
                yaml_rel_path=yaml_rel,
                yaml_content=content_by_rel_path[yaml_rel],
                twig_rel_path=twig_rel,
                twig_content=content_by_rel_path[twig_rel],
            )

    output_file = sft_dir / "combined.jsonl"
    generator.save(output_file)

    manifest.set_metrics({"total_samples": len(generator.samples)})
    manifest.add_output("combined_sft", "sft/combined.jsonl", calculate_hash(output_file))
    manifest.save()

    logger.info(f"SFT generation complete. Generated {len(generator.samples)} samples.")
    return 0
