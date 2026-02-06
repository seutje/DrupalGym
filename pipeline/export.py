import shutil
import subprocess
import sys
from pathlib import Path
import json

import torch
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
            if "gguf" in export_cfg.get("formats", []):
                quantize_to_gguf(output_path, export_cfg, logger)
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


def _resolve_tool_path(configured_path: str | None, candidates: list[str]) -> str | None:
    if configured_path:
        tool = Path(configured_path).expanduser()
        if tool.exists():
            return str(tool)

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


def _run_command(command: list[str], logger: PipelineLogger, fail_message: str):
    rendered = " ".join(command)
    logger.info(f"Running command: {rendered}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.info(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(f"{fail_message}: exit code {result.returncode}")


def quantize_to_gguf(exported_model_dir: Path, export_cfg: dict, logger: PipelineLogger):
    if not exported_model_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {exported_model_dir}")

    gguf_quants = export_cfg.get("quantization", {}).get("gguf", [])
    if not gguf_quants:
        logger.info("No GGUF quantization targets configured. Skipping.")
        return

    tool_cfg = export_cfg.get("tools", {})
    llama_cpp_dir = tool_cfg.get("llama_cpp_dir")

    convert_candidates = ["convert_hf_to_gguf.py"]
    quantize_candidates = ["llama-quantize", "quantize"]

    if llama_cpp_dir:
        convert_candidates.insert(0, str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py"))
        quantize_candidates.insert(0, str(Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"))
        quantize_candidates.insert(1, str(Path(llama_cpp_dir) / "build" / "bin" / "quantize"))

    convert_script = _resolve_tool_path(tool_cfg.get("convert_hf_to_gguf"), convert_candidates)
    if not convert_script:
        raise RuntimeError(
            "Unable to locate convert_hf_to_gguf.py. Configure export.tools.convert_hf_to_gguf "
            "or export.tools.llama_cpp_dir in pipeline.yaml."
        )

    quantize_bin = _resolve_tool_path(tool_cfg.get("llama_quantize"), quantize_candidates)
    if not quantize_bin:
        raise RuntimeError(
            "Unable to locate llama-quantize. Configure export.tools.llama_quantize "
            "or export.tools.llama_cpp_dir in pipeline.yaml."
        )

    _normalize_tokenizer_config_for_conversion(exported_model_dir, logger)

    f16_file = exported_model_dir / "model-f16.gguf"
    convert_command = [
        sys.executable,
        convert_script,
        str(exported_model_dir),
        "--outfile",
        str(f16_file),
        "--outtype",
        "f16",
    ]
    _run_command(convert_command, logger, "GGUF conversion failed")

    generated = [f16_file.name]
    for quant in gguf_quants:
        quant_file = exported_model_dir / f"model-{quant.lower()}.gguf"
        quant_command = [
            quantize_bin,
            str(f16_file),
            str(quant_file),
            quant,
        ]
        _run_command(quant_command, logger, f"GGUF quantization failed for {quant}")
        generated.append(quant_file.name)

    logger.info(
        f"Generated GGUF files in {exported_model_dir}: {', '.join(generated)}",
        gguf_files=generated,
    )


def _normalize_tokenizer_config_for_conversion(exported_model_dir: Path, logger: PipelineLogger):
    tokenizer_config_path = exported_model_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return

    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    extra_special_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return

    if "additional_special_tokens" not in config:
        config["additional_special_tokens"] = extra_special_tokens
    config.pop("extra_special_tokens", None)

    with tokenizer_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    logger.info(
        "Normalized tokenizer_config.json for GGUF conversion compatibility",
        tokenizer_config=str(tokenizer_config_path),
    )
