import argparse
import sys
from pathlib import Path
from .logger import PipelineLogger
from .utils import load_config, get_project_root

def run_stage(stage_name: str, config: dict):
    root = get_project_root()
    logger = PipelineLogger(stage_name, root / "manifests")
    logger.info(f"Starting stage: {stage_name}")
    
    # Placeholder for stage execution logic
    if stage_name == "0":
        logger.info("Phase 0 already in progress via scaffold script.")
    else:
        logger.error(f"Stage {stage_name} not yet implemented.")
        return 1
        
    logger.info(f"Completed stage: {stage_name}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="DrupalGym Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    run_parser = subparsers.add_parser("run", help="Run a pipeline stage")
    run_parser.add_argument("stage", help="Stage name or number to run")
    run_parser.add_argument("--config", default="pipeline.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.command == "run":
        config = load_config(args.config)
        sys.exit(run_stage(args.stage, config))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
