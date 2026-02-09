import json
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Pattern

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash


SYMBOL_PROMPT_RE = re.compile(
    r"^Show me the implementation of the (class|interface|trait|enum) ([A-Za-z_][A-Za-z0-9_]*) in the file (.+)\.$"
)
PROMPT_WRAPPER_RE = re.compile(r"(?mi)^\s*(instruction|input|output)\s*:")
NUMERIC_LINE_RE = re.compile(r"^\d{1,5}(?:[.):])?$")
FENCED_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", re.DOTALL)
SPECIAL_TOKEN_ARTIFACT_RE = re.compile(r"<\|[^|\n]{1,100}\|>")
PROCEDURAL_EXTENSIONS = (".module", ".install", ".inc", ".theme", ".profile")
WHITESPACE_RE = re.compile(r"\s+")
ROOT_PROCEDURAL_PHP = {
    "index.php",
    "update.php",
    "autoload.php",
    ".ht.router.php",
    "authorize.php",
    "cron.php",
    "rebuild.php",
}
DEFAULT_WEAK_CATEGORY_PATTERNS: dict[str, list[str]] = {
    "attributes": [r"#\[[A-Za-z_\\][A-Za-z0-9_\\]*"],
    "di": [
        r"\bContainerInterface\b",
        r"public\s+static\s+function\s+create\s*\(",
        r"services\.yml",
        r"logger\.factory",
        r"__construct\s*\(",
    ],
}
DEFAULT_MODERN_DRUPAL_REQUIRED_CATEGORIES = ("attributes", "di")
DEFAULT_MODERN_DRUPAL_RELEVANT_TYPES = (
    "code_reference",
    "sdc_reference",
    "bugfix",
    "refactor",
    "write_from_spec",
    "explain_and_implement",
)
DEFAULT_MODERN_DRUPAL_SOURCE_PATTERNS = ("/plugin/", ".module", ".install", ".theme", ".services.yml")
DEFAULT_MODERN_DRUPAL_INSTRUCTION_TERMS = (
    "plugin",
    "module",
    "block",
    "service",
    "dependency injection",
    "constructor injection",
    "containerinterface",
)
PHPSTAN_SYNTAX_ERROR_RE = re.compile(r"(syntax error|parse error)", re.IGNORECASE)


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * p)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def _compile_pattern_map(configured: dict | None) -> dict[str, list[Pattern[str]]]:
    source = configured or DEFAULT_WEAK_CATEGORY_PATTERNS
    compiled: dict[str, list[Pattern[str]]] = {}
    for category, values in source.items():
        patterns: list[Pattern[str]] = []
        for raw in values:
            try:
                patterns.append(re.compile(str(raw), re.IGNORECASE | re.MULTILINE))
            except re.error:
                continue
        if patterns:
            compiled[str(category)] = patterns
    return compiled


class QualityGate:
    def __init__(self, logger: PipelineLogger, config: dict | None = None):
        self.logger = logger
        cfg = config or {}
        self.min_output_chars = int(cfg.get("min_output_chars", 150))
        self.max_output_chars = int(cfg.get("max_output_chars", 50000))
        self.min_output_chars_by_type = {
            str(sample_type): int(limit)
            for sample_type, limit in (cfg.get("min_output_chars_by_type", {}) or {}).items()
        }
        self.max_output_chars_by_type = {
            str(sample_type): int(limit)
            for sample_type, limit in (cfg.get("max_output_chars_by_type", {}) or {}).items()
        }
        self.run_php_lint = bool(cfg.get("run_php_lint", False))
        self.php_bin = shutil.which("php") if self.run_php_lint else None
        self.run_phpcs = bool(cfg.get("run_phpcs", False))
        self.phpcs_bin = shutil.which("phpcs") if self.run_phpcs else None
        self.phpcs_drupal_standard_available = self._has_drupal_phpcs_standard() if self.phpcs_bin else False
        self.phpcs_runtime_broken = False
        self.run_phpstan = bool(cfg.get("run_phpstan", False))
        self.phpstan_bin = shutil.which("phpstan") if self.run_phpstan else None
        self.phpstan_failure_mode = str(cfg.get("phpstan_failure_mode", "syntax_only")).strip().lower()
        if self.phpstan_failure_mode not in {"syntax_only", "strict"}:
            self.phpstan_failure_mode = "syntax_only"
        self.runtime_check_workers = max(1, min(3, int(cfg.get("runtime_check_workers", 3))))
        self.reject_prompt_wrapper_echo = bool(cfg.get("reject_prompt_wrapper_echo", True))
        self.reject_path_leakage_tokens = bool(cfg.get("reject_path_leakage_tokens", True))
        self.path_leakage_tokens = [str(token).lower() for token in cfg.get("path_leakage_tokens", ["repos/"])]
        self.max_numeric_line_streak = int(cfg.get("max_numeric_line_streak", 40))
        self.max_repeated_line_ratio = float(cfg.get("max_repeated_line_ratio", 0.25))
        self.duplicate_output_mode = str(cfg.get("duplicate_output_mode", "exact")).strip().lower()
        if self.duplicate_output_mode not in {"exact", "normalized"}:
            self.duplicate_output_mode = "exact"
        self.reject_ambiguous_instruction_input = bool(cfg.get("reject_ambiguous_instruction_input", True))
        self.max_outputs_per_instruction_input = int(cfg.get("max_outputs_per_instruction_input", 1))
        self.require_non_empty_input_for_types = {
            str(sample_type).strip()
            for sample_type in cfg.get("require_non_empty_input_for_types", [])
            if str(sample_type).strip()
        }
        self.doc_source_allowlist_prefixes = [
            str(prefix).strip().lower()
            for prefix in cfg.get("doc_source_allowlist_prefixes", [])
            if str(prefix).strip()
        ]
        self.doc_topic_denylist_terms = [
            str(term).strip().lower()
            for term in cfg.get("doc_topic_denylist_terms", [])
            if str(term).strip()
        ]
        self.enforce_modern_drupal_patterns = bool(cfg.get("enforce_modern_drupal_patterns", False))
        self.modern_drupal_required_categories = {
            str(category).strip()
            for category in cfg.get("modern_drupal_required_categories", DEFAULT_MODERN_DRUPAL_REQUIRED_CATEGORIES)
            if str(category).strip()
        }
        self.modern_drupal_relevant_types = {
            str(sample_type).strip()
            for sample_type in cfg.get("modern_drupal_relevant_types", DEFAULT_MODERN_DRUPAL_RELEVANT_TYPES)
            if str(sample_type).strip()
        }
        self.modern_drupal_source_patterns = [
            str(pattern).strip().lower()
            for pattern in cfg.get("modern_drupal_relevant_source_patterns", DEFAULT_MODERN_DRUPAL_SOURCE_PATTERNS)
            if str(pattern).strip()
        ]
        self.modern_drupal_instruction_terms = [
            str(term).strip().lower()
            for term in cfg.get("modern_drupal_relevant_instruction_terms", DEFAULT_MODERN_DRUPAL_INSTRUCTION_TERMS)
            if str(term).strip()
        ]
        self.weak_category_patterns = _compile_pattern_map(cfg.get("weak_category_patterns"))

        self.rejected_count = 0
        self.passed_count = 0
        self.reasons: dict[str, int] = {}
        self.rejections_by_type: dict[str, int] = {}
        self.rejection_reasons_by_type: dict[str, dict[str, int]] = {}
        self.passed_output_lengths: list[int] = []
        self.seen_output_hashes: set[str] = set()
        self.instruction_input_outputs: dict[tuple[str, str], set[str]] = {}

    def _effective_min_chars(self, sample_type: str) -> int:
        return int(self.min_output_chars_by_type.get(sample_type, self.min_output_chars))

    def _effective_max_chars(self, sample_type: str) -> int:
        return int(self.max_output_chars_by_type.get(sample_type, self.max_output_chars))

    @staticmethod
    def _allows_procedural_php_without_namespace(source: str) -> bool:
        source_lower = source.lower()
        base_name = Path(source_lower).name
        if source_lower.endswith(PROCEDURAL_EXTENSIONS):
            return True
        return base_name in ROOT_PROCEDURAL_PHP

    @staticmethod
    def _numeric_line_streak(output: str) -> int:
        max_streak = 0
        current = 0
        for line in output.splitlines():
            if NUMERIC_LINE_RE.match(line.strip()):
                current += 1
                if current > max_streak:
                    max_streak = current
            else:
                current = 0
        return max_streak

    @staticmethod
    def _repeated_line_ratio(output: str) -> float:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if len(lines) < 20:
            return 0.0
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        max_count = max(counts.values(), default=0)
        return max_count / len(lines) if lines else 0.0

    @staticmethod
    def _has_predominantly_numeric_fenced_block(output: str) -> bool:
        for match in FENCED_BLOCK_RE.finditer(output):
            block = match.group(1)
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 6:
                continue
            numeric_lines = sum(1 for line in lines if NUMERIC_LINE_RE.match(line))
            if numeric_lines / len(lines) >= 0.8:
                return True
        return False

    @staticmethod
    def _has_special_token_artifact(output: str) -> bool:
        if SPECIAL_TOKEN_ARTIFACT_RE.search(output):
            return True
        return "_closed_prs" in output.lower()

    @staticmethod
    def _normalize_for_hash(content: str) -> str:
        return WHITESPACE_RE.sub(" ", content).strip()

    def _output_hash(self, output: str) -> str:
        payload = output
        if self.duplicate_output_mode == "normalized":
            payload = self._normalize_for_hash(output)
        return __import__("hashlib").sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _doc_source_allowed(self, source: str) -> bool:
        if not self.doc_source_allowlist_prefixes:
            return True
        source_lower = source.lower()
        return any(source_lower.startswith(prefix) for prefix in self.doc_source_allowlist_prefixes)

    @staticmethod
    def _with_php_tag(output: str) -> str:
        return output if output.lstrip().startswith("<?php") else f"<?php\n{output}"

    @staticmethod
    def _phpcs_temp_filename_from_source(source: str) -> str:
        raw_name = Path(str(source or "").strip()).name
        if not raw_name:
            return "sample.php"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)
        if not safe_name or safe_name in {".", ".."}:
            return "sample.php"
        return safe_name

    def _write_temp_php(self, output: str) -> str:
        content = self._with_php_tag(output)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as handle:
            handle.write(content)
            return handle.name

    def _php_lint_ok(self, output: str) -> bool:
        if not self.run_php_lint or not self.php_bin:
            return True
        if "<?php" not in output:
            return True

        temp_path = self._write_temp_php(output)

        try:
            proc = subprocess.run(
                [self.php_bin, "-l", temp_path],
                check=False,
                capture_output=True,
                text=True,
            )
            return proc.returncode == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _has_drupal_phpcs_standard(self) -> bool:
        if not self.phpcs_bin:
            return False
        proc = subprocess.run([self.phpcs_bin, "-i"], check=False, capture_output=True, text=True)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return "Drupal" in output

    @staticmethod
    def _phpcs_runtime_misconfigured(output: str) -> bool:
        lower = output.lower()
        return (
            "referenced sniff" in lower and "does not exist" in lower
        ) or "coding standard \"drupal\" is not installed" in lower

    def _phpcs_ok(self, output: str, source: str) -> bool:
        if (
            not self.run_phpcs
            or not self.phpcs_bin
            or not self.phpcs_drupal_standard_available
            or self.phpcs_runtime_broken
        ):
            return True
        if "<?php" not in output:
            return True

        content = self._with_php_tag(output)
        file_name = self._phpcs_temp_filename_from_source(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = str(Path(temp_dir) / file_name)
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(content)
            proc = subprocess.run(
                [self.phpcs_bin, "--standard=Drupal", temp_path],
                check=False,
                capture_output=True,
                text=True,
            )
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if proc.returncode != 0 and self._phpcs_runtime_misconfigured(combined):
                self.phpcs_runtime_broken = True
                self.phpcs_drupal_standard_available = False
                self.logger.error(
                    "PHPCS runtime is misconfigured; disabling PHPCS gate for remaining samples.",
                    error=combined[:500],
                )
                return True
            return proc.returncode == 0

    @staticmethod
    def _extract_phpstan_messages(stdout: str, stderr: str) -> list[str]:
        messages: list[str] = []
        payload = (stdout or "").strip()
        if payload:
            try:
                data = json.loads(payload)
                for value in data.get("errors", []):
                    text = str(value).strip()
                    if text:
                        messages.append(text)
                for file_info in data.get("files", {}).values():
                    for item in file_info.get("messages", []):
                        text = str(item.get("message", "")).strip()
                        if text:
                            messages.append(text)
            except json.JSONDecodeError:
                for line in payload.splitlines():
                    text = line.strip()
                    if text:
                        messages.append(text)
        if not messages:
            for line in (stderr or "").splitlines():
                text = line.strip()
                if text:
                    messages.append(text)
        return messages

    def _phpstan_ok(self, output: str) -> bool:
        if not self.run_phpstan or not self.phpstan_bin:
            return True
        if "<?php" not in output:
            return True

        temp_path = self._write_temp_php(output)
        try:
            proc = subprocess.run(
                [
                    self.phpstan_bin,
                    "analyse",
                    "--no-progress",
                    "--error-format=json",
                    "--level=0",
                    temp_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        if proc.returncode == 0:
            return True

        messages = self._extract_phpstan_messages(proc.stdout, proc.stderr)
        syntax_errors = sum(1 for message in messages if PHPSTAN_SYNTAX_ERROR_RE.search(message))
        if self.phpstan_failure_mode == "syntax_only":
            return syntax_errors == 0
        return False

    def _run_runtime_checks(self, output: str, source: str) -> tuple[bool, str]:
        if "<?php" not in output:
            return True, ""

        checks: list[tuple[str, callable]] = [
            ("php_syntax_error", lambda: self._php_lint_ok(output)),
            ("phpcs_drupal_violation", lambda: self._phpcs_ok(output, source=source)),
            ("phpstan_failure", lambda: self._phpstan_ok(output)),
        ]

        if self.runtime_check_workers <= 1:
            for reason, runner in checks:
                if not runner():
                    return False, reason
            return True, ""

        active_checks: list[tuple[str, callable]] = []
        for reason, runner in checks:
            if reason == "php_syntax_error" and (not self.run_php_lint or not self.php_bin):
                continue
            if reason == "phpcs_drupal_violation" and (
                not self.run_phpcs
                or not self.phpcs_bin
                or not self.phpcs_drupal_standard_available
                or self.phpcs_runtime_broken
            ):
                continue
            if reason == "phpstan_failure" and (not self.run_phpstan or not self.phpstan_bin):
                continue
            active_checks.append((reason, runner))

        if len(active_checks) <= 1:
            for reason, runner in checks:
                if not runner():
                    return False, reason
            return True, ""

        results: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=min(self.runtime_check_workers, len(active_checks))) as executor:
            future_to_reason = {executor.submit(runner): reason for reason, runner in active_checks}
            for future, reason in future_to_reason.items():
                try:
                    results[reason] = bool(future.result())
                except Exception as exc:
                    self.logger.error("Runtime quality check failed.", reason=reason, error=str(exc))
                    results[reason] = False

        for reason, _runner in checks:
            if reason in results and not results[reason]:
                return False, reason
        return True, ""

    def _sample_requires_modern_patterns(
        self, sample_type: str, source: str, instruction_lower: str, output_lower: str
    ) -> bool:
        source_lower = source.lower()
        is_php_like = (
            "<?php" in output_lower
            or source_lower.endswith((".php", ".module", ".install", ".inc", ".theme", ".profile"))
        )
        if not is_php_like:
            return False
        if sample_type in self.modern_drupal_relevant_types:
            return True
        if any(token in source_lower for token in self.modern_drupal_source_patterns):
            return True
        return any(term in instruction_lower for term in self.modern_drupal_instruction_terms)

    def _has_required_modern_pattern(self, sample: dict) -> bool:
        patterns: list[Pattern[str]] = []
        for category in self.modern_drupal_required_categories:
            patterns.extend(self.weak_category_patterns.get(category, []))
        if not patterns:
            return True

        blob = "\n".join(
            [
                str(sample.get("metadata", {}).get("source", "")),
                str(sample.get("instruction", "")),
                str(sample.get("input", "")),
                str(sample.get("output", "")),
            ]
        )
        return any(pattern.search(blob) for pattern in patterns)

    def check_sample(self, sample: dict) -> tuple[bool, str]:
        output = sample.get("output", "")
        instruction = sample.get("instruction", "")
        input_text = str(sample.get("input", ""))
        instruction_lower = instruction.lower()
        sample_type = str(sample.get("metadata", {}).get("type", "unknown") or "unknown")
        source = str(sample.get("metadata", {}).get("source", ""))
        output_hash = self._output_hash(output)

        pair_key = (instruction.strip(), input_text.strip())
        if self.reject_ambiguous_instruction_input and self.max_outputs_per_instruction_input > 0:
            seen_outputs = self.instruction_input_outputs.get(pair_key, set())
            if output_hash not in seen_outputs and len(seen_outputs) >= self.max_outputs_per_instruction_input:
                return False, "ambiguous_instruction_input_pair"

        if sample_type in self.require_non_empty_input_for_types and not input_text.strip():
            return False, "missing_context_input"

        if sample_type == "yaml_reference":
            if "yaml configuration" not in instruction_lower:
                return False, "yaml_instruction_output_mismatch"
            if ":" not in output or "<?php" in output:
                return False, "yaml_instruction_output_mismatch"

        if sample_type == "doc_summary":
            if not instruction_lower.startswith("explain the following topic"):
                return False, "doc_instruction_output_mismatch"
            alpha_char_count = sum(1 for char in output if char.isalpha())
            if alpha_char_count < 80:
                return False, "doc_instruction_output_mismatch"
            topic = str(sample.get("metadata", {}).get("topic", "")).strip().lower()
            doc_blob = "\n".join([topic, instruction_lower, output.lower()])
            if any(term in doc_blob for term in self.doc_topic_denylist_terms):
                return False, "doc_topic_denied"
            if not self._doc_source_allowed(source):
                return False, "doc_source_not_allowed"

        if self.enforce_modern_drupal_patterns and self._sample_requires_modern_patterns(
            sample_type=sample_type,
            source=source,
            instruction_lower=instruction_lower,
            output_lower=output.lower(),
        ):
            if not self._has_required_modern_pattern(sample):
                return False, "missing_drupal11_attribute_or_di_pattern"

        if len(output) < self._effective_min_chars(sample_type):
            return False, "too_short"
        if len(output) > self._effective_max_chars(sample_type):
            return False, "too_long"
        if self.reject_prompt_wrapper_echo and PROMPT_WRAPPER_RE.search(output):
            return False, "prompt_wrapper_echo"
        if self.reject_path_leakage_tokens:
            model_facing_text = "\n".join([instruction, str(sample.get("input", "")), output]).lower()
            for token in self.path_leakage_tokens:
                if token and token in model_facing_text:
                    return False, "path_leakage_token"
        if self._has_special_token_artifact(output):
            return False, "special_token_artifact"

        numeric_streak = self._numeric_line_streak(output)
        if numeric_streak >= self.max_numeric_line_streak:
            return False, "numeric_line_streak"

        repeated_ratio = self._repeated_line_ratio(output)
        if repeated_ratio >= self.max_repeated_line_ratio:
            return False, "repetitive_output"
        if self._has_predominantly_numeric_fenced_block(output):
            return False, "numeric_code_block_artifact"

        if output_hash in self.seen_output_hashes:
            return False, "near_duplicate_content"

        boilerplate_terms = ["cookie", "yes, please", "no, do not track me", "sign in", "log in", "create an account"]
        for term in boilerplate_terms:
            if term in output.lower()[:200]:
                return False, "boilerplate_content"

        if instruction.startswith("Show me the implementation of"):
            match = SYMBOL_PROMPT_RE.match(instruction)
            if not match:
                return False, "invalid_symbol_kind_prompt"

        if "explain the following topic" in instruction_lower:
            topic = instruction.split(":")[-1].strip().lower()
            if len(topic) < 6:
                return False, "poor_instruction"
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
            }
            if topic in generic_titles:
                return False, "generic_topic"
            if any(term in topic for term in ["cookie", "web beacon", "sign in"]):
                return False, "irrelevant_topic"

        lines = [line for line in output.split("\n") if line.strip()]
        if len(lines) < 3 and len(output) < 500:
            return False, "insufficient_detail"

        if sample.get("metadata", {}).get("type") == "doc_summary":
            content_lower = output.lower()
            if "drupal 7" in content_lower and not any(
                marker in content_lower for marker in ["drupal 8", "drupal 9", "drupal 10", "drupal 11", "symfony"]
            ):
                return False, "drupal_7_only"

        if (
            "<?php" in output
            and "namespace" not in output
            and "hook_" not in output
            and not self._allows_procedural_php_without_namespace(source)
        ):
            return False, "missing_namespace_in_php"

        runtime_ok, runtime_reason = self._run_runtime_checks(output, source)
        if not runtime_ok:
            return False, runtime_reason

        self.seen_output_hashes.add(output_hash)
        pair_outputs = self.instruction_input_outputs.setdefault(pair_key, set())
        pair_outputs.add(output_hash)

        return True, ""

    def process(self, input_path: Path, output_path: Path, rejected_path: Path):
        with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out, open(
            rejected_path, "w", encoding="utf-8"
        ) as f_rej:
            for line in f_in:
                try:
                    sample = json.loads(line)
                    is_passed, reason = self.check_sample(sample)
                    if is_passed:
                        f_out.write(json.dumps(sample, ensure_ascii=True) + "\n")
                        self.passed_count += 1
                        self.passed_output_lengths.append(len(str(sample.get("output", ""))))
                    else:
                        sample["rejection_reason"] = reason
                        f_rej.write(json.dumps(sample, ensure_ascii=True) + "\n")
                        self.rejected_count += 1
                        self.reasons[reason] = self.reasons.get(reason, 0) + 1
                        sample_type = str(sample.get("metadata", {}).get("type", "unknown") or "unknown")
                        self.rejections_by_type[sample_type] = self.rejections_by_type.get(sample_type, 0) + 1
                        typed_reasons = self.rejection_reasons_by_type.setdefault(sample_type, {})
                        typed_reasons[reason] = typed_reasons.get(reason, 0) + 1
                except Exception as exc:
                    self.logger.error(f"Error in quality gate: {str(exc)}")


def run_quality_stage(config: dict, logger: PipelineLogger, root: Path):
    sft_dir = root / "sft"
    quality_dir = root / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    input_file = sft_dir / "combined.jsonl"
    output_file = quality_dir / "passed.jsonl"
    rejected_file = quality_dir / "rejected.jsonl"

    if not input_file.exists():
        logger.error("sft/combined.jsonl not found.")
        return 1

    manifest = Manifest("quality_gates", quality_dir)
    manifest.add_input("sft_combined", "1.0", calculate_hash(input_file))

    gate_cfg = config.get("quality", {})
    gate = QualityGate(logger, config=gate_cfg)
    gate.process(input_file, output_file, rejected_file)

    report = {
        "passed": gate.passed_count,
        "rejected": gate.rejected_count,
        "rejection_reasons": gate.reasons,
        "rejections_by_type": gate.rejections_by_type,
        "rejection_reasons_by_type": gate.rejection_reasons_by_type,
        "output_length_percentiles": {
            "p50": _percentile(gate.passed_output_lengths, 0.50),
            "p90": _percentile(gate.passed_output_lengths, 0.90),
            "p95": _percentile(gate.passed_output_lengths, 0.95),
            "p99": _percentile(gate.passed_output_lengths, 0.99),
        },
        "pass_rate": gate.passed_count / (gate.passed_count + gate.rejected_count)
        if (gate.passed_count + gate.rejected_count) > 0
        else 0,
    }

    with open(quality_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    manifest.set_metrics(report)
    manifest.add_output("passed_sft", "quality/passed.jsonl", calculate_hash(output_file))
    manifest.add_output("rejected_sft", "quality/rejected.jsonl", calculate_hash(rejected_file))
    manifest.save()

    logger.info(f"Quality gate complete. Report: {report}")
    return 0
