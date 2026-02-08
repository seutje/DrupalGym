import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash


SYMBOL_PROMPT_RE = re.compile(
    r"^Show me the implementation of the (class|interface|trait|enum) ([A-Za-z_][A-Za-z0-9_]*) in the file (.+)\.$"
)
PROMPT_WRAPPER_RE = re.compile(r"(?mi)^\s*(instruction|input|output)\s*:")
NUMERIC_LINE_RE = re.compile(r"^\d{1,5}(?:[.):])?$")
FENCED_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", re.DOTALL)
PROCEDURAL_EXTENSIONS = (".module", ".install", ".inc", ".theme", ".profile")
ROOT_PROCEDURAL_PHP = {
    "index.php",
    "update.php",
    "autoload.php",
    ".ht.router.php",
    "authorize.php",
    "cron.php",
    "rebuild.php",
}


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * p)
    return ordered[max(0, min(len(ordered) - 1, idx))]


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
        self.reject_prompt_wrapper_echo = bool(cfg.get("reject_prompt_wrapper_echo", True))
        self.max_numeric_line_streak = int(cfg.get("max_numeric_line_streak", 40))
        self.max_repeated_line_ratio = float(cfg.get("max_repeated_line_ratio", 0.25))

        self.rejected_count = 0
        self.passed_count = 0
        self.reasons: dict[str, int] = {}
        self.rejections_by_type: dict[str, int] = {}
        self.rejection_reasons_by_type: dict[str, dict[str, int]] = {}
        self.passed_output_lengths: list[int] = []
        self.seen_output_hashes: set[str] = set()

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

    def _php_lint_ok(self, output: str) -> bool:
        if not self.run_php_lint or not self.php_bin:
            return True
        if "<?php" not in output:
            return True

        content = output if output.lstrip().startswith("<?php") else f"<?php\n{output}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as handle:
            handle.write(content)
            temp_path = handle.name

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

    def check_sample(self, sample: dict) -> tuple[bool, str]:
        output = sample.get("output", "")
        instruction = sample.get("instruction", "")
        instruction_lower = instruction.lower()
        sample_type = str(sample.get("metadata", {}).get("type", "unknown") or "unknown")
        source = str(sample.get("metadata", {}).get("source", ""))

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

        if len(output) < self._effective_min_chars(sample_type):
            return False, "too_short"
        if len(output) > self._effective_max_chars(sample_type):
            return False, "too_long"
        if self.reject_prompt_wrapper_echo and PROMPT_WRAPPER_RE.search(output):
            return False, "prompt_wrapper_echo"

        numeric_streak = self._numeric_line_streak(output)
        if numeric_streak >= self.max_numeric_line_streak:
            return False, "numeric_line_streak"

        repeated_ratio = self._repeated_line_ratio(output)
        if repeated_ratio >= self.max_repeated_line_ratio:
            return False, "repetitive_output"
        if self._has_predominantly_numeric_fenced_block(output):
            return False, "numeric_code_block_artifact"

        output_hash = __import__("hashlib").sha256(output.encode("utf-8", errors="ignore")).hexdigest()
        if output_hash in self.seen_output_hashes:
            return False, "near_duplicate_content"
        self.seen_output_hashes.add(output_hash)

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

        if not self._php_lint_ok(output):
            return False, "php_syntax_error"

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
