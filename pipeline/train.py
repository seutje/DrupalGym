import os
import json
import re
import copy
import shutil
import subprocess
import tempfile
import traceback
import torch
from typing import Any

def patch_torch_matmul():
    if getattr(torch, "matmul_orig", None):
        return
    torch.matmul_orig = torch.matmul
    def patched_matmul(a, b, out=None):
        if a.is_cuda and out is None:
            if a.dim() == 2 and b.dim() == 2:
                return torch.mm(a, b)
            
            # Handle the case where one is 2D and the other is >2D
            if a.dim() > 2 and b.dim() == 2:
                a_shape = a.shape
                res = torch.mm(a.reshape(-1, a_shape[-1]), b)
                return res.reshape(*a_shape[:-1], b.shape[-1])
            
            if a.dim() == 2 and b.dim() > 2:
                # b is (Batch, N, P), a is (M, N)
                # This is less common but can happen
                b_shape = b.shape
                # We want (M, N) @ (Batch, N, P) -> (Batch, M, P)
                # torch.matmul does this by broadcasting a
                res = torch.mm(a, b.transpose(0, 1).reshape(b_shape[1], -1))
                return res.reshape(a.shape[0], b_shape[0], b_shape[2]).transpose(0, 1)

            if a.dim() >= 3 and b.dim() >= 3:
                # Batched matmul with both side batched
                # If they have same batch dims, we can loop
                if a.shape[:-2] == b.shape[:-2]:
                    if a.dim() == 3:
                        res = torch.empty(a.shape[0], a.shape[1], b.shape[2], device=a.device, dtype=a.dtype)
                        for i in range(a.shape[0]):
                            res[i] = torch.mm(a[i], b[i])
                        return res
                    if a.dim() == 4:
                        res = torch.empty(a.shape[0], a.shape[1], a.shape[2], b.shape[3], device=a.device, dtype=a.dtype)
                        for i in range(a.shape[0]):
                            for j in range(a.shape[1]):
                                res[i, j] = torch.mm(a[i, j], b[i, j])
                        return res
        
        return torch.matmul_orig(a, b, out=out)
    torch.matmul = patched_matmul

patch_torch_matmul()

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


def _build_model_config_with_compat_fixes(
    *,
    model_name: str,
    logger: PipelineLogger,
    auto_config_cls,
    mistral3_config_cls=None,
    config_dict_loader=None,
):
    try:
        return auto_config_cls.from_pretrained(model_name, trust_remote_code=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        exc_str = str(exc).lower()
        if "ministral3" not in exc_str and "mistral3" not in exc_str:
            raise

        if mistral3_config_cls is None:
            try:
                from transformers import MinistralConfig
                mistral3_config_cls = MinistralConfig
                target_type = "ministral"
            except ImportError:
                try:
                    from transformers import MistralConfig
                    mistral3_config_cls = MistralConfig
                    target_type = "mistral"
                except ImportError:
                    try:
                        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
                        mistral3_config_cls = Mistral3Config
                        target_type = "mistral"
                    except Exception:
                        raise

        if config_dict_loader is not None:
            config_data = config_dict_loader(model_name)
        else:
            # Safely try to get the config dict without relying on missing AutoConfig methods
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(repo_id=model_name, filename="config.json")
                with open(config_path, "r", encoding="utf-8") as handle:
                    config_data = json.load(handle)
            except Exception as loader_exc:
                # Fallback to internal transformers method if available
                get_config_dict = getattr(auto_config_cls, "get_config_dict", None)
                if callable(get_config_dict):
                    config_data, _ = get_config_dict(model_name, trust_remote_code=True)
                else:
                    raise RuntimeError(
                        f"Unable to load raw model config for Ministral compatibility patch: {loader_exc}"
                    ) from loader_exc

        patched = copy.deepcopy(config_data)
        
        # Remove pre-existing quantization_config to avoid conflicts with our BitsAndBytesConfig
        if "quantization_config" in patched:
            logger.info("Removing pre-existing quantization_config from model config to avoid conflicts.")
            del patched["quantization_config"]

        # Patch top-level model_type if it's mistral3/ministral3
        if str(patched.get("model_type", "")).lower() in ("ministral3", "mistral3"):
            patched["model_type"] = target_type

        # Patch text_config if present (common in hierarchical configs)
        text_cfg = patched.get("text_config")
        if isinstance(text_cfg, dict):
            if str(text_cfg.get("model_type", "")).lower() in ("ministral3", "mistral3"):
                text_cfg["model_type"] = "mistral"
            
            # Promote key parameters to top-level for compatibility with CausalLM classes
            for key in ["hidden_size", "num_attention_heads", "num_hidden_layers", 
                        "num_key_value_heads", "head_dim", "rms_norm_eps", 
                        "intermediate_size", "max_position_embeddings", "vocab_size"]:
                if key in text_cfg and key not in patched:
                    patched[key] = text_cfg[key]
            
            from transformers import MistralConfig
            patched["text_config"] = MistralConfig.from_dict(text_cfg)

        # Patch vision_config if present
        vision_cfg = patched.get("vision_config")
        if isinstance(vision_cfg, dict):
            from transformers import PixtralVisionConfig
            patched["vision_config"] = PixtralVisionConfig.from_dict(vision_cfg)

        logger.info(
            f"Applied compatibility patch for Ministral-3 using {mistral3_config_cls.__name__}.",
            model=model_name,
            target_type=target_type
        )

        return mistral3_config_cls.from_dict(patched)


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
            "Upgrade `transformers` to a version that supports Ministral-3 and retry stage 7."
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


def _build_completion_labels(token_ids: list[int], marker_tokens: list[int]) -> list[int]:
    labels = list(token_ids)
    marker_index = _find_subsequence(token_ids, marker_tokens)
    if marker_index >= 0:
        response_start = marker_index + len(marker_tokens)
        for idx in range(response_start):
            labels[idx] = -100
        return labels
    return [-100] * len(labels)


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
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = model_config["base_model"]
    logger.info(f"Starting training for {model_name}")

    # 1. Load Dataset
    data_files = {
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "valid.jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)

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
    marker_tokens = tokenizer(completion_marker, add_special_tokens=False)["input_ids"]
    if not marker_tokens:
        raise ValueError(f"Unable to tokenize completion marker: {completion_marker!r}")
    logger.info(
        "Using training prompt template.",
        prompt_template=prompt_template,
        completion_marker=completion_marker,
    )

    def tokenize_function(examples):
        texts = [
            _format_training_text(ins, inp, out, prompt_template=prompt_template)
            for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        tokenized = tokenizer(texts, truncation=True, max_length=train_cfg["max_seq_len"])
        tokenized["labels"] = [_build_completion_labels(token_ids, marker_tokens) for token_ids in tokenized["input_ids"]]
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
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
    
    model_config_obj = _build_model_config_with_compat_fixes(
        model_name=model_name,
        logger=logger,
        auto_config_cls=AutoConfig,
    )

    # Patch state dict and config to handle prefix mismatch and FP8 artifacts
    import tempfile
    from safetensors.torch import load_file, save_file
    from huggingface_hub import snapshot_download

    model_load_path = model_name
    tmp_dir_obj = None

    try:
        snapshot_dir = snapshot_download(model_name)
        weight_path = Path(snapshot_dir) / "model.safetensors"
        
        if weight_path.exists():
            tmp_dir_obj = tempfile.TemporaryDirectory()
            tmp_dir = Path(tmp_dir_obj.name)
            
            logger.info(f"Creating patched model directory at {tmp_dir}")
            
            # 1. Patch and save weights
            raw_state_dict = load_file(str(weight_path))
            patched_state_dict = {}
            prefix = "language_model."
            converted_count = 0
            for k, v in raw_state_dict.items():
                if k.endswith((".activation_scale", ".weight_scale_inv")):
                    continue
                
                # Convert FP8 to BF16 to allow bitsandbytes 4-bit quantization
                if "float8" in str(v.dtype):
                    v = v.to(torch.bfloat16)
                    converted_count += 1
                
                if k.startswith(prefix):
                    patched_state_dict[k[len(prefix):]] = v
                else:
                    patched_state_dict[k] = v
            
            logger.info(f"Converted {converted_count} FP8 tensors to BF16.")
            logger.info(f"Saving patched weights to {tmp_dir / 'model.safetensors'}...")
            save_file(patched_state_dict, str(tmp_dir / "model.safetensors"))
            logger.info("Weights saved successfully.")
            del raw_state_dict
            del patched_state_dict
            
            # 2. Save patched config
            logger.info("Saving patched config...")
            model_config_obj.save_pretrained(str(tmp_dir))
            logger.info("Config saved successfully.")
            
            model_load_path = str(tmp_dir)
        else:
            logger.info("Local model.safetensors not found; falling back to default loading.")
    except Exception as e:
        logger.info(f"Model patching failed or skipped: {e}")

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            config=model_config_obj,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    except Exception as exc:
        _raise_actionable_model_load_error(exc, model_name)
    finally:
        if tmp_dir_obj:
            try:
                tmp_dir_obj.cleanup()
            except Exception:
                pass

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
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": train_cfg["use_reentrant_gc"]},
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
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
