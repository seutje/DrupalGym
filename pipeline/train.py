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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard"
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
    dataset_version = "v1"
    dataset_dir = root / "dataset" / dataset_version
    models_dir = root / "models"
    
    for model_cfg in config.get("models", []):
        model_name = model_cfg["name"]
        output_dir = models_dir / model_name / "test_run"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Preparing training for {model_name}...")
        # In a real environment, we would call train_model here
        # For this execution, we will skip the actual training call
        # but log that it's "ready".
        logger.info(f"SKIPPING actual training for {model_name} (No GPU environment)")
        
    return 0
