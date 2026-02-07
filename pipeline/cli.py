import argparse
import sys
from .logger import PipelineLogger
from .sources import run_sources_stage
from .acquisition import run_acquisition_stage
from .normalization import run_normalization_stage
from .sft_generation import run_sft_generation_stage
from .quality_gates import run_quality_stage
from .dataset_packaging import run_packaging_stage
from .dataset_refinement import run_dataset_refinement_stage
from .train import run_training_stage
from .evaluation import run_evaluation_stage
from .utils import load_config, get_project_root

def run_stage(stage_name: str, config: dict):
    root = get_project_root()
    logger = PipelineLogger(stage_name, root / "manifests")
    logger.info(f"Starting stage: {stage_name}")
    
    if stage_name in {"0", "phase0"}:
        logger.info("Phase 0 already in progress via scaffold script.")
    elif stage_name in {"1", "phase1", "sources"}:
        return run_sources_stage(config, logger, root)
    elif stage_name in {"2", "phase2", "acquisition"}:
        return run_acquisition_stage(config, logger, root)
    elif stage_name in {"3", "phase3", "normalization"}:
        return run_normalization_stage(config, logger, root)
    elif stage_name in {"4", "phase4", "sft"}:
        return run_sft_generation_stage(config, logger, root)
    elif stage_name in {"5", "phase5", "quality"}:
        return run_quality_stage(config, logger, root)
    elif stage_name in {"6", "phase6", "dataset"}:
        return run_packaging_stage(config, logger, root)
    elif stage_name in {"6b", "phase6b", "dataset_refine", "refine"}:
        return run_dataset_refinement_stage(config, logger, root)
    elif stage_name in {"7", "phase7", "train"}:
        return run_training_stage(config, logger, root, mode="test_run")
    elif stage_name in {"8", "phase8", "eval"}:
        return run_evaluation_stage(config, logger, root)
    elif stage_name in {"9", "phase9", "full_train"}:
        return run_training_stage(config, logger, root, mode="full_scale")
    elif stage_name in {"10", "phase10", "export"}:
        from .export import run_export_stage
        return run_export_stage(config, logger, root)
    elif stage_name in {"11", "phase11", "hardening"}:
        logger.info("Phase 11: Automation and reproducibility hardening.")
        return 0
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
    
    # Add a pipeline runner command
    subparsers.add_parser("pipeline", help="Run the full pipeline")
    
    args = parser.parse_args()
    
    if args.command == "run":
        config = load_config(args.config)
        sys.exit(run_stage(args.stage, config))
    elif args.command == "pipeline":
        from .runner import run_pipeline
        run_pipeline()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
