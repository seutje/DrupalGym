import json
import os
import random
from pathlib import Path
from .manifest import Manifest, calculate_hash
from .logger import PipelineLogger

class InstructionGenerator:
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.samples = []

    def generate_from_php(self, content: str, rel_path: str):
        """Simple template-based generation from PHP code."""
        # This is a placeholder for more sophisticated logic
        # e.g., parsing class names, hook names, etc.
        
        # Example: if it's a plugin
        if 'namespace Drupal\\' in content and 'class ' in content:
            class_match = re.search(r'class\s+(\w+)', content)
            if class_match:
                class_name = class_match.group(1)
                self.samples.append({
                    "instruction": f"Show me the implementation of the class {class_name} in the file {rel_path}.",
                    "input": "",
                    "output": content,
                    "metadata": {
                        "source": rel_path,
                        "type": "code_reference"
                    }
                })

    def generate_from_doc(self, content: str, rel_path: str):
        """Generate instructions from documentation."""
        lines = content.split('\n')
        title = lines[0] if lines else "Drupal 11 Documentation"
        
        self.samples.append({
            "instruction": f"Explain the following topic based on Drupal 11 documentation: {title}",
            "input": "",
            "output": content,
            "metadata": {
                "source": rel_path,
                "type": "doc_summary"
            }
        })

    def save(self, output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + '\n')

import re

def run_sft_generation_stage(config: dict, logger: PipelineLogger, root: Path):
    clean_dir = root / "clean"
    sft_dir = root / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = Manifest("sft_generation", sft_dir)
    
    generator = InstructionGenerator(logger)
    
    # Process cleaned files
    for dirpath, _, filenames in os.walk(clean_dir):
        for filename in filenames:
            if filename == "dedup_manifest.json":
                continue
                
            clean_path = Path(dirpath) / filename
            rel_path = str(clean_path.relative_to(clean_dir))
            
            try:
                with open(clean_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if clean_path.suffix.lower() in ['.php', '.module', '.inc']:
                    generator.generate_from_php(content, rel_path)
                elif clean_path.suffix.lower() in ['.html', '.md', '.txt']:
                    generator.generate_from_doc(content, rel_path)
            except Exception as e:
                logger.error(f"Error generating SFT samples for {clean_path}: {str(e)}")

    output_file = sft_dir / "combined.jsonl"
    generator.save(output_file)
    
    manifest.set_metrics({
        "total_samples": len(generator.samples)
    })
    manifest.add_output("combined_sft", "sft/combined.jsonl", calculate_hash(output_file))
    manifest.save()
    
    logger.info(f"SFT generation complete. Generated {len(generator.samples)} samples.")
    return 0
