import sys
from .cli import run_stage
from .utils import load_config, get_project_root
from .logger import PipelineLogger

def run_pipeline(stages=None):
    config = load_config()
    root = get_project_root()
    
    if not stages:
        stages = ["sources", "acquisition", "normalization", "sft", "quality", "dataset", "train", "eval"]
    
    for stage in stages:
        print(f"\n>>> Running stage: {stage}")
        result = run_stage(stage, config)
        if result != 0:
            print(f"!!! Stage {stage} failed with code {result}")
            sys.exit(result)
            
    print("\n>>> Pipeline execution complete!")

if __name__ == "__main__":
    run_pipeline()
