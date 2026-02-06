import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .logger import PipelineLogger

def run_export_stage(config: dict, logger: PipelineLogger, root: Path):
    export_cfg = config.get("export", {})
    models_dir = root / "models"
    
    # We look for final models first, then test_run models
    models_to_export = []
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        final_adapter = model_dir / "final" / "adapter"
        test_adapter = model_dir / "test_run" / "adapter"
        
        if final_adapter.exists():
            models_to_export.append((model_dir.name, "final", final_adapter))
        elif test_adapter.exists():
            models_to_export.append((model_dir.name, "test_run", test_adapter))
            
    if not models_to_export:
        logger.error("No adapters found to export.")
        return 1
        
    for model_name, suffix, adapter_path in models_to_export:
        logger.info(f"Exporting {model_name} ({suffix})...")
        
        # Get base model name from adapter config or pipeline config
        # For simplicity, we'll try to find it in the pipeline config
        base_model_name = None
        for m in config.get("models", []):
            if m["name"] == model_name:
                base_model_name = m["base_model"]
                break
        
        if not base_model_name:
            # Fallback for test models not in main models list
            if "training" in config and "test_run" in config["training"]:
                for m in config["training"]["test_run"].get("models", []):
                    if m["name"] == model_name:
                        base_model_name = m["base_model"]
                        break
        
        if not base_model_name:
            logger.error(f"Could not determine base model for {model_name}")
            continue
            
        output_path = adapter_path.parent / "exported"
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            export_model(base_model_name, adapter_path, output_path, export_cfg, logger)
        except Exception as e:
            logger.error(f"Export failed for {model_name}: {str(e)}")
            
    return 0

def export_model(base_model_name: str, adapter_path: Path, output_path: Path, export_cfg: dict, logger: PipelineLogger):
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load base model in FP16/BF16 for merging
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="cpu", # Merge on CPU to save VRAM if needed, or "auto"
        trust_remote_code=True
    )
    
    logger.info(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    if export_cfg.get("merge_adapters", True):
        logger.info("Merging adapter into base model...")
        model = model.merge_and_unload()
        
    logger.info(f"Saving exported model to {output_path}")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    
    logger.info(f"Exported to {output_path}")
    
    # TODO: Implement GGUF quantization if requested
    if "gguf" in export_cfg.get("formats", []):
        logger.info("GGUF format requested. Note: GGUF conversion requires llama.cpp scripts.")
        # We could potentially call a shell script here if llama.cpp is present
