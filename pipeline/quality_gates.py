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
        
        # 1. Basic length checks
        if len(output) < 50:
            return False, "too_short"
        if len(output) > 50000:
            return False, "too_long"
            
        # 2. Drupal 11 Specificity (Heuristic)
        # Check for PHP 8 attributes in code samples
        if sample.get("metadata", {}).get("type") == "code_reference":
            if "#[" in output:
                # Good: contains attributes
                pass
            elif "/**" in output and "@" in output:
                # Potential old-style annotations. 
                # We might not reject them yet, but we could flag them.
                pass
        
        # 3. Quality indicators
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
