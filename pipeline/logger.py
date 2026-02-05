import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

class PipelineLogger:
    """Standardized logging and metrics format (JSONL)."""
    
    def __init__(self, stage: str, log_dir: Path):
        self.stage = stage
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{stage}_run.jsonl"

    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": self.stage,
            "level": level,
            "message": message,
        }
        if extra:
            entry.update(extra)
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def info(self, message: str, **kwargs):
        self.log("INFO", message, kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, kwargs)

    def metric(self, name: str, value: Any, **kwargs):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": self.stage,
            "type": "metric",
            "name": name,
            "value": value,
        }
        entry.update(kwargs)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
