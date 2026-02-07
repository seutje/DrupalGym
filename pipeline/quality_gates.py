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


class QualityGate:
    def __init__(self, logger: PipelineLogger, config: dict | None = None):
        self.logger = logger
        cfg = config or {}
        self.min_output_chars = int(cfg.get("min_output_chars", 150))
        self.max_output_chars = int(cfg.get("max_output_chars", 50000))
        self.run_php_lint = bool(cfg.get("run_php_lint", False))
        self.php_bin = shutil.which("php") if self.run_php_lint else None
        self.reject_prompt_wrapper_echo = bool(cfg.get("reject_prompt_wrapper_echo", True))
        self.max_numeric_line_streak = int(cfg.get("max_numeric_line_streak", 40))
        self.max_repeated_line_ratio = float(cfg.get("max_repeated_line_ratio", 0.25))

        self.rejected_count = 0
        self.passed_count = 0
        self.reasons: dict[str, int] = {}
        self.seen_output_hashes: set[str] = set()

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

        if len(output) < self.min_output_chars:
            return False, "too_short"
        if len(output) > self.max_output_chars:
            return False, "too_long"
        if self.reject_prompt_wrapper_echo and PROMPT_WRAPPER_RE.search(output):
            return False, "prompt_wrapper_echo"

        numeric_streak = self._numeric_line_streak(output)
        if numeric_streak >= self.max_numeric_line_streak:
            return False, "numeric_line_streak"

        repeated_ratio = self._repeated_line_ratio(output)
        if repeated_ratio >= self.max_repeated_line_ratio:
            return False, "repetitive_output"

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

        if "<?php" in output and "namespace" not in output and "hook_" not in output:
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
                    else:
                        sample["rejection_reason"] = reason
                        f_rej.write(json.dumps(sample, ensure_ascii=True) + "\n")
                        self.rejected_count += 1
                        self.reasons[reason] = self.reasons.get(reason, 0) + 1
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
