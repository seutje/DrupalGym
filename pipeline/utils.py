import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root."""
    return Path(__file__).parent.parent.absolute()

def load_config(config_path: str = "pipeline.yaml"):
    """Loads the pipeline configuration file."""
    root = get_project_root()
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)
