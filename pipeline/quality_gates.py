import json
import re
from pathlib import Path
from .manifest import Manifest, calculate_hash
from .logger import PipelineLogger

class QualityGate:
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.rejected_count = 0
        self.passed_count = 0
        self.reasons = {}

    def check_sample(self, sample: dict) -> tuple[bool, str]:
        output = sample.get("output", "")
        instruction = sample.get("instruction", "").lower()
        
        # 1. Basic length checks
        if len(output) < 150: # Increased minimum length
            return False, "too_short"
        if len(output) > 50000:
            return False, "too_long"
            
        # 2. Boilerplate Detection
        boilerplate_terms = ["cookie", "yes, please", "no, do not track me", "sign in", "log in", "create an account"]
        for term in boilerplate_terms:
            if term in output.lower()[:200]: # Check beginning of file
                return False, "boilerplate_content"

        # 3. Instruction quality
        if "explain the following topic" in instruction:
             topic = instruction.split(":")[-1].strip()
             if len(topic) < 6:
                 return False, "poor_instruction"
             
             generic_titles = [
                "contents of this file", "introduction", "readme", "license", 
                "requirements", "installation", "configuration", "for developers",
                "description", "features", "support", "author", "maintainers",
                "copyright", "how it works", "prerequisites", "cors configuration",
                "tracking script verification", "using the condition", "cookie behavior",
                "gnu general public license"
             ]
             if topic in generic_titles:
                 return False, "generic_topic"

             if any(term in topic for term in ["cookie", "web beacon", "sign in"]):
                 return False, "irrelevant_topic"

        # 4. Content diversity check (avoid single-word or path-only outputs)
        lines = [l for l in output.split('\n') if l.strip()]
        if len(lines) < 3 and len(output) < 500:
             return False, "insufficient_detail"

        # 5. Drupal 11 / Modernity check
        # If it's a doc summary, it should ideally mention modern Drupal or at least not be exclusively D7
        if sample.get("metadata", {}).get("type") == "doc_summary":
            content_lower = output.lower()
            if "drupal 7" in content_lower and not any(v in content_lower for v in ["drupal 8", "drupal 9", "drupal 10", "drupal 11", "symfony"]):
                return False, "drupal_7_only"

        # 5. PHP Quality
        if "<?php" in output and "namespace" not in output:
            if "hook_" not in output:
                return False, "missing_namespace_in_php"

        return True, ""

    def process(self, input_path: Path, output_path: Path, rejected_path: Path):
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out, \
             open(rejected_path, 'w', encoding='utf-8') as f_rej:
            
            for line in f_in:
                try:
                    sample = json.loads(line)
                    is_passed, reason = self.check_sample(sample)
                    
                    if is_passed:
                        f_out.write(json.dumps(sample) + '\n')
                        self.passed_count += 1
                    else:
                        sample["rejection_reason"] = reason
                        f_rej.write(json.dumps(sample) + '\n')
                        self.rejected_count += 1
                        self.reasons[reason] = self.reasons.get(reason, 0) + 1
                except Exception as e:
                    self.logger.error(f"Error in quality gate: {str(e)}")

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

    gate = QualityGate(logger)
    gate.process(input_file, output_file, rejected_file)
    
    report = {
        "passed": gate.passed_count,
        "rejected": gate.rejected_count,
        "rejection_reasons": gate.reasons,
        "pass_rate": gate.passed_count / (gate.passed_count + gate.rejected_count) if (gate.passed_count + gate.rejected_count) > 0 else 0
    }
    
    with open(quality_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    manifest.set_metrics(report)
    manifest.add_output("passed_sft", "quality/passed.jsonl", calculate_hash(output_file))
    manifest.save()
    
    logger.info(f"Quality gate complete. Report: {report}")
    return 0
