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

def train_model(model_config: dict, dataset_dir: Path, output_dir: Path, logger: PipelineLogger):
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
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        texts = [f"Instruction: {ins}\nInput: {inp}\nOutput: {out}" 
                 for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=2048)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # 3. Model Configuration (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1, # Reduced for 12GB VRAM
        gradient_accumulation_steps=16, # Compensate for small batch size
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=10, # Short test run
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        gradient_checkpointing=True, # Essential for 12GB VRAM
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
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

    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Detected GPU: {gpu_name}")

    dataset_version = "v1"
    dataset_dir = root / "dataset" / dataset_version
    models_dir = root / "models"
    
    # For testing on 4070, we'll process the first model in the config
    # to avoid running out of disk space or time if multiple are defined.
    models_to_train = config.get("models", [])
    if not models_to_train:
        logger.error("No models defined in configuration.")
        return 1

    for model_cfg in models_to_train:
        model_name = model_cfg["name"]
        output_dir = models_dir / model_name / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting actual training for {model_name}...")
        try:
            train_model(model_cfg, dataset_dir, output_dir, logger)
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {str(e)}")
            return 1
        
    return 0
