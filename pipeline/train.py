import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .logger import PipelineLogger

def _resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float16

def train_model(
    model_config: dict,
    dataset_dir: Path,
    output_dir: Path,
    logger: PipelineLogger,
    train_cfg: dict,
):
    model_name = model_config["base_model"]
    logger.info(f"Starting training for {model_name}")

    # 1. Load Dataset
    data_files = {
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "valid.jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        texts = [f"Instruction: {ins}\nInput: {inp}\nOutput: {out}" 
                 for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
        # Use dynamic padding instead of max_length to save VRAM
        return tokenizer(texts, truncation=True, max_length=train_cfg["max_seq_len"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # 3. Model Configuration (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_resolve_dtype(train_cfg["bnb_4bit_compute_dtype"])
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["lora_target_modules"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=1,
        max_steps=train_cfg["max_steps"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": train_cfg["use_reentrant_gc"]},
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        group_by_length=train_cfg["group_by_length"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 5. Execute Training
    trainer.train()
    
    # 6. Save Adapter
    model.save_pretrained(str(output_dir / "adapter"))
    logger.info(f"Training completed for {model_name}. Adapter saved to {output_dir / 'adapter'}")

def run_training_stage(config: dict, logger: PipelineLogger, root: Path):
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. GPU is required for training.")
        return 1

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"Detected GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")

    dataset_version = "v1"
    dataset_dir = root / "dataset" / dataset_version
    models_dir = root / "models"
    
    default_test_cfg = {
        "max_seq_len": 512,
        "max_steps": 5,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "no",
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "use_reentrant_gc": False,
        "group_by_length": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "bnb_4bit_compute_dtype": "float16",
        "max_models": 1,
    }
    train_cfg = default_test_cfg | config.get("training", {}).get("test_run", {})
    
    models_to_train = train_cfg.get("models", config.get("models", []))
    if not models_to_train:
        logger.error("No models defined in configuration.")
        return 1
    models_to_train = models_to_train[: int(train_cfg["max_models"])]

    for model_cfg in models_to_train:
        torch.cuda.empty_cache()
        model_name = model_cfg["name"]
        output_dir = models_dir / model_name / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting actual training for {model_name}...")
        try:
            train_model(model_cfg, dataset_dir, output_dir, logger, train_cfg)
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            return 1
        
    return 0
