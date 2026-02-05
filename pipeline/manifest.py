import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

class Manifest:
    """Handles versioned manifest files for pipeline stages."""
    
    def __init__(self, stage: str, output_dir: Path):
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.data = {
            "stage": stage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputs": [],
            "outputs": {},
            "metrics": {},
            "config": {}
        }

    def add_input(self, name: str, version: str, hash_val: Optional[str] = None):
        self.data["inputs"].append({
            "name": name,
            "version": version,
            "hash": hash_val
        })

    def add_output(self, key: str, path: str, hash_val: str):
        self.data["outputs"][key] = {
            "path": path,
            "hash": hash_val
        }

    def set_metrics(self, metrics: Dict[str, Any]):
        self.data["metrics"].update(metrics)

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = self.output_dir / "manifest.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        return path

def calculate_hash(file_path: Path) -> str:
    """Calculates SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
