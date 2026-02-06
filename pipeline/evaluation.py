import json
import torch
import traceback
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from .logger import PipelineLogger

PROMPT_SUITE = [
    {
        "id": "block_attribute",
        "category": "attributes",
        "instruction": "Create a Drupal 11 Block plugin using PHP 8 attributes. The block ID should be 'gym_stats' and the label 'Gym Statistics'.",
    },
    {
        "id": "service_di",
        "category": "di",
        "instruction": "Define a Drupal 11 service in gym.services.yml and its class implementation using constructor injection for the 'logger.factory' service.",
    },
    {
        "id": "routing_sdc",
        "category": "sdc",
        "instruction": "How do I define a Single Directory Component (SDC) in Drupal 11? Show a basic directory structure and the component.yml file.",
    }
]

def generate_response(model, tokenizer, instruction, max_new_tokens=512):
    prompt = f"Instruction: {instruction}\nInput: \nOutput: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def run_automated_checks(prompt_id, output):
    checks = {
        "length": len(output),
        "has_php_tag": "<?php" in output,
    }
    
    if prompt_id == "block_attribute":
        checks["use_attributes"] = "#[Block" in output
        checks["has_id"] = "id: 'gym_stats'" in output or 'id: "gym_stats"' in output
        checks["has_label"] = "label: 'Gym Statistics'" in output or 'label: "Gym Statistics"' in output
    elif prompt_id == "service_di":
        checks["has_services_yml"] = "services:" in output
        checks["has_injection"] = "__construct" in output and ("LoggerInterface" in output or "logger.factory" in output)
    elif prompt_id == "routing_sdc":
        checks["has_component_yml"] = "component.yml" in output
        checks["has_directory_structure"] = "components/" in output
        
    return checks

def run_evaluation_stage(config: dict, logger: PipelineLogger, root: Path):
    try:
        eval_dir = root / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        sample_outputs_dir = eval_dir / "sample_outputs"
        sample_outputs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting evaluation stage...")
        
        # 1. Determine model to evaluate
        train_cfg = config.get("training", {}).get("test_run", {})
        models_to_eval = train_cfg.get("models", config.get("models", []))
        if not models_to_eval:
            logger.error("No models defined in configuration.")
            return 1
        
        model_cfg = models_to_eval[0]
        model_name = model_cfg["name"]
        base_model_id = model_cfg["base_model"]
        adapter_path = root / "models" / model_name / "test_run" / "adapter"
        
        if not adapter_path.exists():
            logger.error(f"Adapter not found at {adapter_path}. Did stage 7 complete?")
            return 1

        logger.info(f"Loading model {base_model_id} on CPU (stable fallback)...")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Using CPU for stability in this environment
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        
        logger.info("Loading adapter...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        # 3. Evaluation Loop
        results = []
        for prompt in PROMPT_SUITE:
            logger.info(f"Evaluating fine-tuned {prompt['id']}...")
            output = generate_response(model, tokenizer, prompt["instruction"])
            metrics = run_automated_checks(prompt["id"], output)
            
            results.append({
                "model": model_name,
                "type": "fine-tuned",
                "prompt": prompt,
                "generated_output": output,
                "metrics": metrics
            })
            
            output_file = sample_outputs_dir / f"{model_name}_{prompt['id']}.txt"
            with open(output_file, "w") as f:
                f.write(output)
            logger.info(f"Saved output to {output_file}")

        # 4. Baseline Evaluation
        logger.info("Running baseline evaluation (disabling adapter)...")
        with model.disable_adapter():
            for prompt in PROMPT_SUITE:
                logger.info(f"Evaluating baseline {prompt['id']}...")
                output = generate_response(model, tokenizer, prompt["instruction"])
                metrics = run_automated_checks(prompt["id"], output)
                
                results.append({
                    "model": base_model_id,
                    "type": "baseline",
                    "prompt": prompt,
                    "generated_output": output,
                    "metrics": metrics
                })
                
                output_file = sample_outputs_dir / f"baseline_{prompt['id']}.txt"
                with open(output_file, "w") as f:
                    f.write(output)
                logger.info(f"Saved baseline output to {output_file}")

        # 5. Save Results
        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Evaluation complete. Results saved to {metrics_path}")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
