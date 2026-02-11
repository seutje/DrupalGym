import os
import json
import re
import shutil
import subprocess
import tempfile
import traceback
import torch
from typing import Any

from pathlib import Path
from .logger import PipelineLogger

NUMERIC_LINE_RE = re.compile(r"^\d{1,5}(?:[.):])?$")
FENCED_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", re.DOTALL)
SPECIAL_TOKEN_ARTIFACT_RE = re.compile(r"<\|[^|\n]{1,100}\|>")
FIM_MARKER_RE = re.compile(
    r"(?i)(<\|fim_(prefix|middle|suffix|pad)\|>|<fim_(prefix|middle|suffix|pad)>|<\|file_sep\|>)"
)
PROMPT_WRAPPER_RE = re.compile(r"(?mi)^\s*(instruction|input|output|response|assistant|user)\s*:")
MALFORMED_WRAPPER_RE = re.compile(
    r"(?im)(\[\s*/?inst\s*\]|^\s*###\s*(instruction|input|output|response)\s*:|<\|im_(start|end)\|>|<\|assistant\|>|<\|user\|>)"
)
OUTPUT_MARKER = "Output:"
PROMPT_TEMPLATE_PLAIN = "instruction_input_output"
PROMPT_TEMPLATE_MINISTRAL_INST = "ministral_inst"

def _resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    modules = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "mistral_common": "mistral_common",
    }
    for key, module_name in modules.items():
        try:
            module = __import__(module_name)
            versions[key] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[key] = "unavailable"
    return versions


def _ensure_native_ministral3_support(model_name: str):
    if "ministral-3" not in model_name.lower():
        return
    try:
        import transformers

        version = str(getattr(transformers, "__version__", "unknown"))
    except Exception:
        version = "unknown"
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError(
            f"Unable to verify native Ministral-3 support for transformers={version}. "
            "Install transformers>=5.0.0 and retry stage 7."
        )
    major = int(match.group(1))
    if major < 5:
        raise RuntimeError(
            f"Ministral-3 requires transformers>=5.0.0; detected transformers={version}. "
            "Install a supported transformers version and retry stage 7."
        )


def _normalize_ministral3_config_for_causallm(model_name: str, config_obj, logger: PipelineLogger):
    if "ministral-3" not in model_name.lower():
        return config_obj

    class_name = type(config_obj).__name__
    if class_name != "Mistral3Config":
        return config_obj

    try:
        from transformers import Ministral3Config
    except Exception:
        return config_obj

    logger.info(
        "Normalizing Mistral3Config to Ministral3Config for AutoModelForCausalLM compatibility.",
        model=model_name,
    )
    return Ministral3Config.from_dict(config_obj.to_dict())


def _strip_pretrained_quantization_config(config_obj, logger: PipelineLogger):
    config_dict_getter = getattr(config_obj, "to_dict", None)
    if not callable(config_dict_getter):
        return config_obj
    config_data = config_dict_getter()
    if not isinstance(config_data, dict) or "quantization_config" not in config_data:
        return config_obj

    logger.info("Removing pre-existing quantization_config from model config to avoid conflicts.")
    config_data = dict(config_data)
    config_data.pop("quantization_config", None)
    config_cls = type(config_obj)
    from_dict = getattr(config_cls, "from_dict", None)
    if callable(from_dict):
        try:
            return from_dict(config_data)
        except Exception:
            return config_obj
    return config_obj


def _coerce_model_config_object(config_obj, logger: PipelineLogger):
    if not isinstance(config_obj, dict):
        return config_obj
    model_type = str(config_obj.get("model_type", "")).strip().lower()
    try:
        if model_type == "ministral3":
            from transformers import Ministral3Config

            rebuilt = Ministral3Config.from_dict(config_obj)
        else:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            config_cls = CONFIG_MAPPING[model_type]
            rebuilt = config_cls.from_dict(config_obj)
    except Exception as exc:
        raise RuntimeError(
            f"Model config unexpectedly became a dict for model_type={model_type or 'unknown'} "
            "and could not be rebuilt into a config object."
        ) from exc

    logger.info(
        "Rebuilt dict model config into config object.",
        model_type=model_type or "unknown",
        rebuilt_type=type(rebuilt).__name__,
    )
    return rebuilt


def _raise_actionable_model_load_error(exc: Exception, model_name: str):
    if isinstance(exc, KeyError) and str(exc).strip("'\"") == "ministral3":
        try:
            import transformers

            version = getattr(transformers, "__version__", "unknown")
        except Exception:
            version = "unknown"
        raise RuntimeError(
            "Model load failed because the installed `transformers` build does not recognize "
            f"model_type `ministral3` for {model_name} (detected transformers={version}). "
            "Install transformers>=5.0.0 and retry stage 7."
        ) from exc
    message = str(exc).lower()
    if "ministral3" in message or "mistral3" in message:
        try:
            import transformers

            version = getattr(transformers, "__version__", "unknown")
        except Exception:
            version = "unknown"
        raise RuntimeError(
            f"Model load failed for {model_name} without native Ministral-3 support "
            f"(detected transformers={version}). Install transformers>=5.0.0 and retry stage 7."
        ) from exc
    raise exc


def _load_tokenizer_for_model(
    *,
    model_name: str,
    logger: PipelineLogger,
    auto_tokenizer_cls,
    mistral_tokenizer_cls=None,
):
    try:
        tokenizer = auto_tokenizer_cls.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as exc:
        error_text = str(exc)
        if "Tokenizer class TokenizersBackend does not exist or is not currently imported." not in error_text:
            raise

        if mistral_tokenizer_cls is None:
            try:
                from transformers import MistralCommonTokenizer

                mistral_tokenizer_cls = MistralCommonTokenizer
            except Exception as import_exc:
                raise RuntimeError(
                    "Failed to load tokenizer via AutoTokenizer due to a TokenizersBackend class mismatch. "
                    "Install or upgrade `mistral-common` and `transformers`, then retry stage 7."
                ) from import_exc

        logger.info(
            "AutoTokenizer failed with TokenizersBackend mismatch; using MistralCommonTokenizer fallback.",
            model=model_name,
        )
        try:
            tokenizer = mistral_tokenizer_cls.from_pretrained(model_name, trust_remote_code=True)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Tokenizer fallback with MistralCommonTokenizer failed. "
                "Upgrade `mistral-common` and `transformers`, then retry stage 7."
            ) from fallback_exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _find_subsequence(tokens: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(tokens):
        return -1
    limit = len(tokens) - len(pattern) + 1
    for idx in range(limit):
        if tokens[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def _build_completion_labels(
    token_ids: list[int], marker_token_variants: list[list[int]]
) -> tuple[list[int], bool]:
    labels = list(token_ids)
    marker_index = -1
    marker_len = 0
    for marker_tokens in marker_token_variants:
        marker_index = _find_subsequence(token_ids, marker_tokens)
        if marker_index >= 0:
            marker_len = len(marker_tokens)
            break

    if marker_index >= 0:
        response_start = marker_index + marker_len
        for idx in range(response_start):
            labels[idx] = -100
        return labels, True
    return labels, False


def _build_completion_marker_variants(tokenizer, completion_marker: str) -> list[list[int]]:
    variants: list[list[int]] = []
    for candidate in (completion_marker, f" {completion_marker}", f"\n{completion_marker}"):
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if token_ids and token_ids not in variants:
            variants.append(token_ids)
    return variants


def _resolve_prompt_template(model_config: dict[str, Any]) -> str:
    configured = str(model_config.get("prompt_template", "")).strip().lower()
    if configured:
        return configured

    base_model = str(model_config.get("base_model", "")).lower()
    model_name = str(model_config.get("name", "")).lower()
    if "ministral-3" in base_model or "ministral-3" in model_name:
        return PROMPT_TEMPLATE_MINISTRAL_INST
    return PROMPT_TEMPLATE_PLAIN


def _completion_marker_for_prompt_template(prompt_template: str) -> str:
    if prompt_template == PROMPT_TEMPLATE_MINISTRAL_INST:
        return "[/INST]"
    return OUTPUT_MARKER


def _format_training_text(
    instruction: str,
    input_text: str,
    output_text: str,
    *,
    prompt_template: str,
) -> str:
    instruction = str(instruction or "")
    input_text = str(input_text or "")
    output_text = str(output_text or "")
    if prompt_template == PROMPT_TEMPLATE_MINISTRAL_INST:
        return f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output_text}</s>"
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _build_completion_data_collator(
    *,
    pad_token_id: int,
    padding_strategy: str = "dynamic",
    pad_to_multiple_of: int | None = None,
    fixed_max_length: int | None = None,
):
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            return {}

        sequence_lengths = [len(feature["input_ids"]) for feature in features]
        if padding_strategy == "fixed_max_length" and fixed_max_length:
            target_length = int(fixed_max_length)
        else:
            target_length = max(sequence_lengths)
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            target_length = _round_up_to_multiple(target_length, int(pad_to_multiple_of))

        pad_values = {
            "input_ids": int(pad_token_id),
            "attention_mask": 0,
            "labels": -100,
            "token_type_ids": 0,
        }
        batch: dict[str, list[list[int]]] = {
            key: [] for key in pad_values if key in features[0]
        }

        for feature in features:
            feature_length = len(feature["input_ids"])
            pad_size = target_length - feature_length
            for key in batch:
                values = list(feature[key])
                batch[key].append(values + [pad_values[key]] * pad_size)

        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}

    return collate


def _has_predominantly_numeric_fenced_block(output: str) -> bool:
    for match in FENCED_BLOCK_RE.finditer(output):
        block = match.group(1)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 6:
            continue
        numeric_lines = sum(1 for line in lines if NUMERIC_LINE_RE.match(line))
        if numeric_lines / len(lines) >= 0.8:
            return True
    return False


def _has_special_token_artifact(output: str) -> bool:
    if SPECIAL_TOKEN_ARTIFACT_RE.search(output):
        return True
    if FIM_MARKER_RE.search(output):
        return True
    return "_closed_prs" in output.lower()


def _has_prompt_wrapper_leakage(output: str) -> bool:
    return bool(PROMPT_WRAPPER_RE.search(output) or MALFORMED_WRAPPER_RE.search(output))


def _artifact_failure_reasons(
    output: str,
    *,
    max_numeric_line_streak: int,
    max_repeated_line_ratio: float,
) -> list[str]:
    reasons: list[str] = []
    if _has_prompt_wrapper_leakage(output):
        reasons.append("prompt_wrapper_echo")
    if _numeric_line_streak(output) > max_numeric_line_streak:
        reasons.append("numeric_line_streak")
    if _repeated_line_ratio(output) > max_repeated_line_ratio:
        reasons.append("repetitive_output")
    if _has_predominantly_numeric_fenced_block(output):
        reasons.append("numeric_code_block_artifact")
    if _has_special_token_artifact(output):
        reasons.append("special_token_artifact")
    return reasons


def _numeric_line_streak(output: str) -> int:
    max_streak = 0
    current_streak = 0
    for line in output.splitlines():
        if NUMERIC_LINE_RE.match(line.strip()):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def _repeated_line_ratio(output: str) -> float:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 20:
        return 0.0
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return max(counts.values()) / len(lines) if counts else 0.0


def _audit_dataset_artifacts(
    *,
    dataset_dir: Path,
    logger: PipelineLogger,
    max_numeric_line_streak: int,
    max_repeated_line_ratio: float,
) -> bool:
    split_names = ("train", "valid")
    summary = {
        "checked_samples": 0,
        "failed_samples": 0,
        "reasons": {
            "prompt_wrapper_echo": 0,
            "numeric_line_streak": 0,
            "repetitive_output": 0,
            "numeric_code_block_artifact": 0,
            "special_token_artifact": 0,
        },
    }
    failing_examples: list[dict[str, Any]] = []

    for split_name in split_names:
        split_path = dataset_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        with open(split_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                output = str(sample.get("output", ""))
                summary["checked_samples"] += 1
                reasons = _artifact_failure_reasons(
                    output,
                    max_numeric_line_streak=max_numeric_line_streak,
                    max_repeated_line_ratio=max_repeated_line_ratio,
                )
                for reason in reasons:
                    summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
                if reasons:
                    summary["failed_samples"] += 1
                    if len(failing_examples) < 20:
                        failing_examples.append(
                            {
                                "split": split_name,
                                "line": line_number,
                                "reason": ",".join(reasons),
                            }
                        )

    if summary["failed_samples"] > 0:
        logger.error(
            "Dataset artifact audit failed.",
            failed_samples=summary["failed_samples"],
            failing_examples=failing_examples,
        )
    logger.info("Dataset artifact audit completed.", **summary)
    return summary["failed_samples"] == 0


def _load_quality_scorecard(dataset_dir: Path, logger: PipelineLogger) -> dict[str, Any] | None:
    scorecard_path = dataset_dir / "quality_scorecard.json"
    if not scorecard_path.exists():
        logger.error(f"Missing required quality scorecard: {scorecard_path}")
        return None

    try:
        with open(scorecard_path, "r", encoding="utf-8") as handle:
            scorecard = json.load(handle)
    except Exception as exc:
        logger.error(f"Failed to read quality scorecard: {exc}")
        return None

    if not bool(scorecard.get("overall_passed", False)):
        failed_checks = [name for name, passed in (scorecard.get("checks", {}) or {}).items() if not bool(passed)]
        logger.error("Quality scorecard checks failed; refusing to train.", failed_checks=failed_checks)
        return None

    logger.info("Quality scorecard gate passed.", scorecard_path=str(scorecard_path))
    return scorecard


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _missing_quality_tools(config: dict) -> list[str]:
    def _resolve_tool_path(tool: str) -> str | None:
        direct = shutil.which(tool)
        if direct:
            return direct
        candidate_dirs: list[str] = []
        composer_home = os.environ.get("COMPOSER_HOME", "").strip()
        if composer_home:
            candidate_dirs.append(str(Path(composer_home) / "vendor" / "bin"))

        home_candidates = {str(Path.home()), "/root"}
        for home in home_candidates:
            candidate_dirs.append(str(Path(home) / ".config" / "composer" / "vendor" / "bin"))
            candidate_dirs.append(str(Path(home) / ".composer" / "vendor" / "bin"))

        seen: set[str] = set()
        for directory in candidate_dirs:
            if not directory or directory in seen:
                continue
            seen.add(directory)
            tool_path = Path(directory) / tool
            if tool_path.is_file() and os.access(tool_path, os.X_OK):
                return str(tool_path)
        return None

    def _tool_available(tool: str) -> bool:
        return _resolve_tool_path(tool) is not None

    def _phpcs_drupal_standard_usable() -> bool:
        phpcs_bin = _resolve_tool_path("phpcs")
        if not phpcs_bin:
            return False
        with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as handle:
            handle.write("<?php\nclass DrupalGymQualityCheck {}\n")
            temp_path = handle.name
        try:
            proc = subprocess.run(
                [phpcs_bin, "--standard=Drupal", temp_path],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

        if proc.returncode == 0:
            return True
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).lower()
        if ("referenced sniff" in combined and "does not exist" in combined) or (
            "coding standard \"drupal\" is not installed" in combined
        ):
            return False
        return True

    required: set[str] = set()
    quality_cfg = config.get("quality", {})
    evaluation_cfg = config.get("evaluation", {})

    if bool(quality_cfg.get("run_php_lint", False)) or bool(evaluation_cfg.get("run_php_lint", False)):
        required.add("php")
    if bool(quality_cfg.get("run_phpcs", False)) or bool(evaluation_cfg.get("run_phpcs", False)):
        required.add("phpcs")
    if bool(quality_cfg.get("run_phpstan", False)) or bool(evaluation_cfg.get("run_phpstan", False)):
        required.add("phpstan")

    missing = [tool for tool in sorted(required) if not _tool_available(tool)]
    if "phpcs" not in missing and "phpcs" in required and not _phpcs_drupal_standard_usable():
        missing.append("phpcs")
    missing = sorted(set(missing))
    return missing

def train_model(
    model_config: dict,
    dataset_dir: Path,
    output_dir: Path,
    logger: PipelineLogger,
    train_cfg: dict,
):
    from datasets import load_dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Mistral3ForConditionalGeneration,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = model_config["base_model"]
    logger.info(f"Starting training for {model_name}")
    _ensure_native_ministral3_support(model_name)

    # 1. Load Dataset
    data_files = {
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "valid.jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)
    overfit_max_train_samples = int(train_cfg.get("overfit_max_train_samples", 0))
    overfit_max_valid_samples = int(train_cfg.get("overfit_max_valid_samples", 0))
    if overfit_max_train_samples > 0:
        train_count = min(overfit_max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(train_count))
        logger.info(
            "Applying overfit probe train subset.",
            selected_train_samples=train_count,
        )
    if overfit_max_valid_samples > 0:
        valid_count = min(overfit_max_valid_samples, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(valid_count))
        logger.info(
            "Applying overfit probe validation subset.",
            selected_valid_samples=valid_count,
        )

    # 2. Tokenizer
    tokenizer = _load_tokenizer_for_model(
        model_name=model_name,
        logger=logger,
        auto_tokenizer_cls=AutoTokenizer,
    )
    
    supported_prompt_templates = {
        PROMPT_TEMPLATE_PLAIN,
        PROMPT_TEMPLATE_MINISTRAL_INST,
    }
    prompt_template = _resolve_prompt_template(model_config)
    if prompt_template not in supported_prompt_templates:
        logger.info(
            "Unsupported prompt template configured; falling back to default template.",
            configured_prompt_template=prompt_template,
            fallback_prompt_template=PROMPT_TEMPLATE_PLAIN,
        )
        prompt_template = PROMPT_TEMPLATE_PLAIN

    completion_marker = _completion_marker_for_prompt_template(prompt_template)
    marker_token_variants = _build_completion_marker_variants(tokenizer, completion_marker)
    if not marker_token_variants:
        raise ValueError(f"Unable to tokenize completion marker: {completion_marker!r}")
    logger.info(
        "Using training prompt template.",
        prompt_template=prompt_template,
        completion_marker=completion_marker,
    )

    marker_miss_tracker = {"missing": 0, "seen": 0}

    def tokenize_function(examples):
        texts = [
            _format_training_text(ins, inp, out, prompt_template=prompt_template)
            for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        tokenized = tokenizer(texts, truncation=True, max_length=train_cfg["max_seq_len"])
        labels: list[list[int]] = []
        for token_ids in tokenized["input_ids"]:
            label_values, found_marker = _build_completion_labels(token_ids, marker_token_variants)
            labels.append(label_values)
            marker_miss_tracker["seen"] += 1
            if not found_marker:
                marker_miss_tracker["missing"] += 1
        tokenized["labels"] = labels
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    if marker_miss_tracker["missing"] > 0:
        logger.info(
            "Completion marker was not found in some tokenized samples; using full-sequence labels for those samples.",
            marker=completion_marker,
            missing_markers=marker_miss_tracker["missing"],
            total_samples=marker_miss_tracker["seen"],
        )
    train_sequence_lengths = [len(token_ids) for token_ids in tokenized_datasets["train"]["input_ids"]]
    if not train_sequence_lengths:
        raise ValueError("No training samples found after tokenization.")
    logger.info(
        "Tokenized train sequence length summary.",
        min_seq_len=min(train_sequence_lengths),
        max_seq_len=max(train_sequence_lengths),
        requested_max_seq_len=int(train_cfg["max_seq_len"]),
    )

    padding_strategy = str(train_cfg.get("padding_strategy", "dynamic"))
    if padding_strategy not in {"dynamic", "fixed_max_length"}:
        logger.info(
            "Unsupported padding strategy configured; falling back to dynamic padding.",
            configured_padding_strategy=padding_strategy,
        )
        padding_strategy = "dynamic"
    pad_to_multiple_of = train_cfg.get("pad_to_multiple_of")
    data_collator = _build_completion_data_collator(
        pad_token_id=int(tokenizer.pad_token_id),
        padding_strategy=padding_strategy,
        pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of else None,
        fixed_max_length=int(train_cfg["max_seq_len"]),
    )
    logger.info(
        "Using completion-aware data collator.",
        padding_strategy=padding_strategy,
        pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of else None,
    )

    # 3. Model Configuration (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_resolve_dtype(train_cfg["bnb_4bit_compute_dtype"])
    )
    
    is_ministral3_model = "ministral-3" in model_name.lower()
    model_config_obj = None
    if is_ministral3_model:
        model_config_obj = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config_obj = _strip_pretrained_quantization_config(model_config_obj, logger)
        model_config_obj = _coerce_model_config_object(model_config_obj, logger)
    else:
        model_config_obj = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config_obj = _strip_pretrained_quantization_config(model_config_obj, logger)
        model_config_obj = _coerce_model_config_object(model_config_obj, logger)

    model = None
    device_map: dict[str, int] | str = "auto"
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        device_map = {"": torch.cuda.current_device()}

    try:
        if is_ministral3_model:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                config=model_config_obj,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config_obj,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    except Exception as exc:
        _raise_actionable_model_load_error(exc, model_name)

    if model is None:
        raise RuntimeError(f"Failed to load model {model_name}")

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
    save_steps = train_cfg.get("save_steps")
    if save_steps is None:
        save_steps = 100 # Safe default if strategy is steps
    max_steps_cfg = int(train_cfg.get("max_steps", -1))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    warmup_steps = 0
    if max_steps_cfg > 0 and warmup_ratio > 0:
        warmup_steps = max(1, int(max_steps_cfg * warmup_ratio))
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg.get("eval_steps"),
        save_strategy=train_cfg["save_strategy"],
        save_steps=save_steps,
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
        report_to="tensorboard",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": train_cfg["use_reentrant_gc"]},
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        logging_nan_inf_filter=bool(train_cfg.get("logging_nan_inf_filter", False)),
        warmup_steps=warmup_steps,
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        group_by_length=train_cfg["group_by_length"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # 5. Execute Training
    trainer.train()
    
    # 6. Save Adapter
    model.save_pretrained(str(output_dir / "adapter"))
    logger.info(f"Training completed for {model_name}. Adapter saved to {output_dir / 'adapter'}")

def run_training_stage(config: dict, logger: PipelineLogger, root: Path, mode: str = "test_run"):
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. GPU is required for training.")
        return 1

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(f"Detected GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
    logger.info("Training dependency versions.", versions=_dependency_versions())

    dataset_version = config.get("dataset", {}).get("training_version", "v1")
    dataset_dir = root / "dataset" / dataset_version
    if not dataset_dir.exists():
        logger.error(f"Dataset version {dataset_version} not found at {dataset_dir}")
        return 1
    if _load_quality_scorecard(dataset_dir, logger) is None:
        return 1
    if mode in {"full_scale", "final"}:
        full_scale_cfg = config.get("training", {}).get("full_scale", {})
        train_path = dataset_dir / "train.jsonl"
        min_train_samples = int(full_scale_cfg.get("min_train_samples", 8000))
        preferred_train_samples = int(full_scale_cfg.get("preferred_train_samples", 20000))
        train_samples = _count_jsonl_records(train_path)
        if train_samples < min_train_samples:
            logger.error(
                "Train split is below full-scale minimum threshold.",
                train_samples=train_samples,
                min_train_samples=min_train_samples,
            )
            return 1
        if train_samples < preferred_train_samples:
            logger.info(
                "Train split meets minimum threshold but is below preferred full-scale size.",
                train_samples=train_samples,
                preferred_train_samples=preferred_train_samples,
            )

        require_quality_tooling = bool(full_scale_cfg.get("require_quality_tooling", True))
        if require_quality_tooling:
            missing_tools = _missing_quality_tools(config)
            if missing_tools:
                logger.error(
                    "Required quality tools are unavailable for full-scale training.",
                    missing_tools=missing_tools,
                )
                return 1
    models_dir = root / "models"
    
    default_cfg = {
        "max_seq_len": 512,
        "max_steps": 5,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "eval_strategy": "no",
        "save_strategy": "no",
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": True,
        "use_reentrant_gc": False,
        "group_by_length": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "bnb_4bit_compute_dtype": "float32",
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_models": 1,
        "padding_strategy": "dynamic",
        "pad_to_multiple_of": 8,
        "overfit_max_train_samples": 0,
        "overfit_max_valid_samples": 0,
    }
    train_cfg = default_cfg | config.get("training", {}).get(mode, {})
    quality_cfg = config.get("quality", {})

    if not _audit_dataset_artifacts(
        dataset_dir=dataset_dir,
        logger=logger,
        max_numeric_line_streak=int(quality_cfg.get("max_numeric_line_streak", 12)),
        max_repeated_line_ratio=float(quality_cfg.get("max_repeated_line_ratio", 0.15)),
    ):
        logger.error("Dataset artifact audit failed; aborting training run.")
        return 1
    
    models_to_train = train_cfg.get("models", config.get("models", []))
    if not models_to_train:
        logger.error("No models defined in configuration.")
        return 1
    
    max_models = int(train_cfg.get("max_models", len(models_to_train)))
    models_to_train = models_to_train[:max_models]

    for model_cfg in models_to_train:
        torch.cuda.empty_cache()
        model_name = model_cfg["name"]
        suffix = "test_run" if mode == "test_run" else "final"
        output_dir = models_dir / model_name / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting {mode} training for {model_name}...")
        try:
            train_model(model_cfg, dataset_dir, output_dir, logger, train_cfg)
        except Exception as e:
            logger.error(
                f"Training failed for {model_name}: {str(e)}",
                exception_type=type(e).__name__,
                exception_repr=repr(e),
                traceback=traceback.format_exc(),
            )
            return 1
        
    return 0
