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
        generic_titles = [
            "contents of this file", "introduction", "readme", "license", 
            "requirements", "installation", "configuration", "for developers",
            "description", "features", "support", "author", "maintainers",
            "copyright", "how it works", "prerequisites", "cors configuration",
            "tracking script verification", "using the condition", "cookie behavior",
            "bulk update user redirect preferences", "troubleshooting",
            "browser-side reset", "server-side reset", "patches details",
            "local libraries", "gnu general public license"
        ]

        # Skip files that are likely not useful for training Drupal 11 logic
        skip_patterns = [
            "LICENSE", "CHANGELOG", "COPYRIGHT", ".cspell", "MAINTAINERS",
            "SECURITY.txt", "DRUPAL_ORG", "fixtures", "node_modules", "vendor"
        ]
        if any(p.lower() in rel_path.lower() for p in skip_patterns):
            return

        # Try to find an H1 title
        title = ""
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        # If title is generic or missing, try to get it from the path
        if not title or title.lower() in generic_titles:
            # Try to get the module name or something meaningful from path
            path_parts = rel_path.split('/')
            if 'repos' in path_parts:
                idx = path_parts.index('repos')
                if len(path_parts) > idx + 1:
                    module_name = path_parts[idx + 1].replace('_', ' ').title()
                    
                    # Special handling for Drupal Core sub-components
                    if module_name == "Drupal Core" and len(path_parts) > idx + 4:
                        # e.g., repos/drupal_core/core/modules/block -> Block
                        sub_component = path_parts[idx + 4].replace('_', ' ').title()
                        module_name = f"{module_name}: {sub_component}"

                    # If it's a README, combine module name with the title if title is not generic
                    if "README" in rel_path:
                        if title and title.lower() not in generic_titles:
                            title = f"{module_name}: {title}"
                        else:
                            title = module_name
                    else:
                        file_stem = Path(rel_path).stem.replace('-', ' ').replace('_', ' ').title()
                        if title and title.lower() not in generic_titles:
                            title = f"{module_name} ({file_stem}): {title}"
                        else:
                            title = f"{module_name}: {file_stem}"

        if not title:
            # Fallback to first line or filename
            lines = [l for l in content.split('\n') if l.strip() and not l.startswith('!')]
            title = lines[0].strip('# ') if lines else Path(rel_path).stem.replace('-', ' ').replace('_', ' ')
        
        # Clean up title from common noise
        title = re.sub(r'\{#.*?\}', '', title).strip()
        
        # Final quality filters for instruction
        if len(title) < 5 or "cookie" in title.lower() or title.lower() in generic_titles:
            return
            
        # Version filter: if title mentions old version but not 11, skip
        if re.search(r'\b(7|8|9|10)\.x\b', title) and '11' not in title:
            return

        instruction = f"Explain the following topic based on Drupal 11 documentation: {title}"
        
        # Simple deduplication within the same stage run
        if any(s['instruction'] == instruction and s['output'] == content for s in self.samples):
            return

        self.samples.append({
            "instruction": instruction,
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
