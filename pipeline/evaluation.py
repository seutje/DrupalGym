import json
import re
import shutil
import subprocess
import tempfile
import time
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash

EVALUATOR_LOGIC_VERSION = "2026-02-11.1"
ARTIFACT_BLOCKLIST_VERSION = "2026-02-10.1"
PROMPT_TEMPLATE_PLAIN = "plain"
PROMPT_TEMPLATE_MINISTRAL_INST = "ministral_inst"
DEFAULT_PROMPT_SUITE = [
    {
        "id": "block_attribute",
        "category": "attributes",
        "instruction": (
            "Create a Drupal 11 Block plugin using PHP 8.3 attributes. "
            "Return one explicit fenced php code block only. "
            "The block ID must be 'gym_stats' and label 'Gym Statistics'."
        ),
        "input": "",
        "requires_php": True,
        "require_fenced_php": True,
        "required_substrings": ["#[Block", "gym_stats", "Gym Statistics"],
    },
    {
        "id": "service_di",
        "category": "di",
        "instruction": (
            "Define a Drupal 11 service in gym.services.yml and its class implementation using constructor "
            "injection for logger.factory. Return exactly two fenced blocks in this order: yaml, then php."
        ),
        "input": "",
        "requires_php": True,
        "require_fenced_php": True,
        "required_substrings": ["services:", "logger.factory", "__construct("],
    },
    {
        "id": "routing_yaml",
        "category": "routing",
        "instruction": (
            "Create a Drupal 11 gym.routing.yml route and a matching controller method for '/gym/stats'. "
            "Return exactly two fenced blocks in this order: yaml, then php."
        ),
        "input": "",
        "requires_php": True,
        "require_fenced_php": True,
        "required_substrings": ["/gym/stats", "_controller"],
    },
    {
        "id": "sdc_component",
        "category": "sdc",
        "instruction": "Show a Drupal 11 Single Directory Component example with directory structure and component.yml.",
        "input": "",
        "requires_php": False,
        "required_substrings": ["component.yml", ".twig"],
    },
]
PROMPT_WRAPPER_RE = re.compile(r"(?mi)^\s*(instruction|input|output|response|assistant|user)\s*:")
MALFORMED_WRAPPER_RE = re.compile(
    r"(?im)(\[\s*/?inst\s*\]|^\s*###\s*(instruction|input|output|response)\s*:|<\|im_(start|end)\|>|<\|assistant\|>|<\|user\|>)"
)
NUMERIC_LINE_RE = re.compile(r"^\d{1,5}(?:[.):])?$")
PHPSTAN_SYNTAX_ERROR_RE = re.compile(r"(syntax error|parse error)", re.IGNORECASE)
SPECIAL_TOKEN_ARTIFACT_RE = re.compile(r"<\|[^|\n]{1,100}\|>")
FIM_MARKER_RE = re.compile(
    r"(?i)(<\|fim_(prefix|middle|suffix|pad)\|>|<fim_(prefix|middle|suffix|pad)>|<\|file_sep\|>)"
)
GENERATION_STOP_MARKER_RE = re.compile(
    r"(?im)(<\|fim_(prefix|middle|suffix|pad)\|>|<fim_(prefix|middle|suffix|pad)>|<\|file_sep\|>|<\|im_(start|end)\|>|<\|assistant\|>|<\|user\|>|^\s*(instruction|input|output|response)\s*:|^\s*###\s*(instruction|input|output|response)\s*:)"
)
FENCED_CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
PHPCS_TEMPFILE_CLASSNAME_RE = re.compile(r"class name doesn't match filename", re.IGNORECASE)
PHPCS_TEMPFILE_NOISE_SOURCES = {
    "PSR1.Classes.ClassDeclaration.InvalidClassName",
}
PROMPTS_REQUIRING_PHP_SNIPPET = {"block_attribute", "service_di", "routing_yaml"}
DEFAULT_GENERATION_BLOCKLIST_STRINGS = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<fim_prefix>",
    "<fim_middle>",
    "<fim_suffix>",
    "<fim_pad>",
    "<|file_sep|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
    "### Instruction:",
    "### Response:",
]
DEFAULT_GENERATION_STOP_REGEX = [
    r"<\|fim_(prefix|middle|suffix|pad)\|>",
    r"<fim_(prefix|middle|suffix|pad)>",
    r"<\|file_sep\|>",
    r"<\|im_(start|end)\|>",
    r"<\|assistant\|>",
    r"<\|user\|>",
    r"(?m)^\s*(instruction|input|output|response)\s*:",
    r"(?m)^\s*###\s*(instruction|input|output|response)\s*:",
]


def _iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    normalized = [str(item).strip() for item in value if str(item).strip()]
    return normalized or list(fallback)


def _build_generation_profile(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    profile = {
        "do_sample": False,
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 512)),
        "min_new_tokens": int(eval_cfg.get("min_new_tokens", 0)),
        "repetition_penalty": float(eval_cfg.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(eval_cfg.get("no_repeat_ngram_size", 0)),
        "generation_blocklist_strings": _string_list(
            eval_cfg.get("generation_blocklist_strings"), DEFAULT_GENERATION_BLOCKLIST_STRINGS
        ),
        "generation_stop_regex": _string_list(eval_cfg.get("generation_stop_regex"), DEFAULT_GENERATION_STOP_REGEX),
        "apply_generation_stop_truncation": bool(eval_cfg.get("apply_generation_stop_truncation", True)),
        "strict_contract_mode": bool(eval_cfg.get("strict_contract_mode", True)),
        "enforce_no_outside_prose_for_php_required": bool(
            eval_cfg.get("enforce_no_outside_prose_for_php_required", True)
        ),
        "artifact_blocklist_version": ARTIFACT_BLOCKLIST_VERSION,
    }
    profile["generation_profile_sha256"] = _hash_json_payload(profile)
    return profile


def _build_evaluator_metadata(eval_cfg: dict[str, Any], generation_profile: dict[str, Any]) -> dict[str, Any]:
    prompt_suite = eval_cfg.get("prompt_suite", [])
    scoring_profile = {
        "semantic_channels": ["required_checks", "php_lint", "phpstan", "artifact_guard"],
        "style_channels": ["phpcs"],
        "semantic_weight": 0.8,
        "style_weight": 0.2,
        "pass_signal": "semantic",
    }
    return {
        "logic_version": EVALUATOR_LOGIC_VERSION,
        "logic_file": "pipeline/evaluation.py",
        "logic_sha256": calculate_hash(Path(__file__)),
        "prompt_suite_size": len(prompt_suite) if isinstance(prompt_suite, list) else 0,
        "prompt_suite_sha256": _hash_json_payload(prompt_suite),
        "generation_profile_sha256": generation_profile.get("generation_profile_sha256", ""),
        "artifact_blocklist_version": generation_profile.get("artifact_blocklist_version", ARTIFACT_BLOCKLIST_VERSION),
        "scoring_profile": scoring_profile,
    }


def _has_prompt_wrapper_leakage(output: str) -> bool:
    return bool(PROMPT_WRAPPER_RE.search(output) or MALFORMED_WRAPPER_RE.search(output))


def _has_special_token_artifact(output: str) -> bool:
    if SPECIAL_TOKEN_ARTIFACT_RE.search(output):
        return True
    if FIM_MARKER_RE.search(output):
        return True
    return "_closed_prs" in output.lower()


def _build_bad_words_ids(tokenizer, blocklist_strings: list[str]) -> list[list[int]]:
    bad_words: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for marker in blocklist_strings:
        try:
            token_ids = tokenizer(marker, add_special_tokens=False).get("input_ids", [])
        except Exception:
            continue
        if not isinstance(token_ids, list) or not token_ids:
            continue
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids or not all(isinstance(token, int) for token in token_ids):
            continue
        key = tuple(token_ids)
        if key in seen:
            continue
        seen.add(key)
        bad_words.append(token_ids)
    return bad_words


def _compile_stop_patterns(patterns: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for raw in patterns:
        try:
            compiled.append((raw, re.compile(raw, re.IGNORECASE | re.MULTILINE)))
        except re.error:
            continue
    return compiled


def _truncate_on_generation_markers(output: str, eval_cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    trimmed = output.strip()
    details = {
        "truncated_on_marker": False,
        "matched_stop_regex": None,
        "matched_text": None,
    }
    if not bool(eval_cfg.get("apply_generation_stop_truncation", True)):
        return trimmed, details

    compiled = eval_cfg.get("_compiled_stop_patterns")
    if not isinstance(compiled, list):
        compiled = _compile_stop_patterns(_string_list(eval_cfg.get("generation_stop_regex"), DEFAULT_GENERATION_STOP_REGEX))

    matches: list[tuple[int, int, str, str]] = []
    for raw, pattern in compiled:
        found = pattern.search(trimmed)
        if found:
            matches.append((found.start(), found.end(), raw, found.group(0)))
    fallback = GENERATION_STOP_MARKER_RE.search(trimmed)
    if fallback:
        matches.append((fallback.start(), fallback.end(), "GENERATION_STOP_MARKER_RE", fallback.group(0)))
    if not matches:
        return trimmed, details

    start, _end, raw, matched = min(matches, key=lambda item: item[0])
    details["truncated_on_marker"] = True
    details["matched_stop_regex"] = raw
    details["matched_text"] = matched[:120]
    return trimmed[:start].rstrip(), details


def _read_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "seed": int(config.get("seed", 42)),
        "mode": "test_run",
        "max_new_tokens": 512,
        "min_new_tokens": 0,
        "device": "auto",
        "max_models": 1,
        "run_php_lint": True,
        "run_phpcs": False,
        "run_phpstan": False,
        "phpstan_failure_mode": "syntax_only",
        "max_code_checks_per_response": 3,
        "php_snippet_policy": "php_only",
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "generation_blocklist_strings": DEFAULT_GENERATION_BLOCKLIST_STRINGS,
        "generation_stop_regex": DEFAULT_GENERATION_STOP_REGEX,
        "apply_generation_stop_truncation": True,
        "strict_contract_mode": True,
        "enforce_no_outside_prose_for_php_required": True,
        "interim_autofencing": {
            "enabled": False,
            "apply_before_contract_checks": True,
            "preserve_raw_output": True,
        },
        "checkpoint_sweep": {
            "enabled": False,
            "step_interval": 0,
            "include_adapter": True,
            "explicit_steps": [],
            "max_checkpoints": 0,
        },
        "prompt_suite": DEFAULT_PROMPT_SUITE,
    }
    merged = defaults | config.get("evaluation", {})
    merged["interim_autofencing"] = defaults["interim_autofencing"] | merged.get("interim_autofencing", {})
    merged["checkpoint_sweep"] = defaults["checkpoint_sweep"] | merged.get("checkpoint_sweep", {})
    prompt_suite = merged.get("prompt_suite") or DEFAULT_PROMPT_SUITE
    normalized_prompts: list[dict[str, Any]] = []
    for item in prompt_suite:
        if not isinstance(item, dict):
            continue
        prompt_id = str(item.get("id", "")).strip()
        instruction = str(item.get("instruction", "")).strip()
        if not prompt_id or not instruction:
            continue
        normalized = dict(item)
        normalized["id"] = prompt_id
        normalized["instruction"] = instruction
        normalized["input"] = str(item.get("input", ""))
        normalized["category"] = str(item.get("category", "general")).strip() or "general"
        normalized["requires_php"] = bool(item.get("requires_php", prompt_id in PROMPTS_REQUIRING_PHP_SNIPPET))
        normalized["require_fenced_php"] = bool(item.get("require_fenced_php", False))
        expected_fenced_blocks = item.get("expected_fenced_blocks")
        normalized["expected_fenced_blocks"] = int(expected_fenced_blocks) if str(expected_fenced_blocks).strip().isdigit() else None
        normalized["expected_fenced_languages"] = [
            str(language).strip().lower()
            for language in item.get("expected_fenced_languages", [])
            if str(language).strip()
        ]
        normalized["enforce_no_outside_prose"] = item.get("enforce_no_outside_prose")
        normalized["required_substrings"] = [str(value) for value in item.get("required_substrings", []) if str(value).strip()]
        normalized["required_regex"] = [str(value) for value in item.get("required_regex", []) if str(value).strip()]
        normalized_prompts.append(normalized)
    merged["prompt_suite"] = normalized_prompts or DEFAULT_PROMPT_SUITE
    merged["seed"] = int(merged.get("seed", defaults["seed"]))
    merged["max_new_tokens"] = int(merged.get("max_new_tokens", defaults["max_new_tokens"]))
    merged["min_new_tokens"] = int(merged.get("min_new_tokens", defaults["min_new_tokens"]))
    merged["max_models"] = int(merged.get("max_models", defaults["max_models"]))
    merged["max_code_checks_per_response"] = int(
        merged.get("max_code_checks_per_response", defaults["max_code_checks_per_response"])
    )
    merged["php_snippet_policy"] = _read_php_snippet_policy(merged)
    merged["repetition_penalty"] = float(merged.get("repetition_penalty", defaults["repetition_penalty"]))
    merged["no_repeat_ngram_size"] = int(merged.get("no_repeat_ngram_size", defaults["no_repeat_ngram_size"]))
    merged["generation_blocklist_strings"] = _string_list(
        merged.get("generation_blocklist_strings"), DEFAULT_GENERATION_BLOCKLIST_STRINGS
    )
    merged["generation_stop_regex"] = _string_list(merged.get("generation_stop_regex"), DEFAULT_GENERATION_STOP_REGEX)
    merged["apply_generation_stop_truncation"] = bool(
        merged.get("apply_generation_stop_truncation", defaults["apply_generation_stop_truncation"])
    )
    merged["strict_contract_mode"] = bool(merged.get("strict_contract_mode", defaults["strict_contract_mode"]))
    merged["enforce_no_outside_prose_for_php_required"] = bool(
        merged.get(
            "enforce_no_outside_prose_for_php_required",
            defaults["enforce_no_outside_prose_for_php_required"],
        )
    )
    return merged


def _resolve_models_for_eval(config: dict[str, Any], eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    configured = eval_cfg.get("models")
    if configured:
        if isinstance(configured, list):
            return [model for model in configured if isinstance(model, dict)]
        return []

    mode = str(eval_cfg.get("mode", "test_run"))
    mode_models = config.get("training", {}).get(mode, {}).get("models")
    if mode_models:
        return [model for model in mode_models if isinstance(model, dict)]

    fallback = config.get("models", [])
    return [model for model in fallback if isinstance(model, dict)]


def _resolve_prompt_template(model_config: dict[str, Any]) -> str:
    configured = str(model_config.get("prompt_template", "")).strip().lower()
    if configured:
        return configured

    base_model = str(model_config.get("base_model", "")).lower()
    model_name = str(model_config.get("name", "")).lower()
    if "ministral-3" in base_model or "ministral-3" in model_name:
        return PROMPT_TEMPLATE_MINISTRAL_INST
    return PROMPT_TEMPLATE_PLAIN


def _build_prompt(instruction: str, input_text: str = "", *, prompt_template: str = PROMPT_TEMPLATE_PLAIN) -> str:
    if prompt_template == PROMPT_TEMPLATE_MINISTRAL_INST:
        return f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput: "


def _model_input_device(model) -> Any:
    try:
        return model.device
    except Exception:
        first_param = next(model.parameters())
        return first_param.device


def _build_generation_kwargs(tokenizer, max_new_tokens: int, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    repetition_penalty = float(eval_cfg.get("repetition_penalty", 1.0))
    if repetition_penalty > 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    no_repeat_ngram_size = int(eval_cfg.get("no_repeat_ngram_size", 0))
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    min_new_tokens = int(eval_cfg.get("min_new_tokens", 0))
    if min_new_tokens > 0:
        kwargs["min_new_tokens"] = min_new_tokens
    bad_words_ids = eval_cfg.get("_bad_words_ids")
    if not isinstance(bad_words_ids, list):
        bad_words_ids = _build_bad_words_ids(
            tokenizer, _string_list(eval_cfg.get("generation_blocklist_strings"), DEFAULT_GENERATION_BLOCKLIST_STRINGS)
        )
    if bad_words_ids:
        kwargs["bad_words_ids"] = bad_words_ids
    return kwargs


def _generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int,
    eval_cfg: dict[str, Any],
    *,
    prompt_template: str = PROMPT_TEMPLATE_PLAIN,
) -> str:
    prompt = _build_prompt(instruction, input_text, prompt_template=prompt_template)
    inputs = None
    if prompt_template == PROMPT_TEMPLATE_MINISTRAL_INST and hasattr(tokenizer, "apply_chat_template"):
        user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
        messages = [{"role": "user", "content": user_content}]
        try:
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(templated, return_tensors="pt")
        except Exception:
            inputs = None
    if inputs is None:
        inputs = tokenizer(prompt, return_tensors="pt")
    input_device = _model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with __import__("torch").no_grad():
        generation_kwargs = _build_generation_kwargs(tokenizer, max_new_tokens=max_new_tokens, eval_cfg=eval_cfg)
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_tokens = inputs["input_ids"][0].tolist()
    output_tokens = outputs[0].tolist()
    if len(output_tokens) >= len(prompt_tokens) and output_tokens[: len(prompt_tokens)] == prompt_tokens:
        response_tokens = output_tokens[len(prompt_tokens) :]
    else:
        # Seq2seq-style generation can return only generated tokens.
        response_tokens = output_tokens
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    stripped = response.strip()
    if stripped:
        return stripped

    # Keep a debuggable payload when the model only emits special tokens/whitespace.
    if response_tokens:
        return tokenizer.decode(response_tokens, skip_special_tokens=False).strip()
    return ""


def _extract_code_blocks(output: str) -> list[str]:
    blocks = [block["code"] for block in _extract_fenced_code_blocks(output)]
    if blocks:
        return blocks
    if "<?php" in output:
        return [output.strip()]
    return []


def _extract_fenced_code_blocks(output: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for match in FENCED_CODE_BLOCK_RE.finditer(output):
        code = match.group(2).strip()
        if not code:
            continue
        blocks.append(
            {
                "language": (match.group(1) or "").strip().lower(),
                "code": code,
            }
        )
    return blocks


def _contract_diagnostics(output: str) -> dict[str, Any]:
    blocks = _extract_fenced_code_blocks(output)
    languages = [block["language"] or "plain" for block in blocks]

    outside_segments: list[str] = []
    cursor = 0
    for match in FENCED_CODE_BLOCK_RE.finditer(output):
        segment = output[cursor : match.start()]
        if segment.strip():
            outside_segments.append(segment.strip())
        cursor = match.end()
    tail = output[cursor:]
    if tail.strip():
        outside_segments.append(tail.strip())

    return {
        "fenced_block_count": len(blocks),
        "fenced_block_languages": languages,
        "outside_prose_detected": len(outside_segments) > 0,
        "outside_prose_excerpt": outside_segments[0][:160] if outside_segments else "",
    }


def _looks_like_yaml(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return False
    sample = lines[:40]
    yaml_like = 0
    for line in sample:
        if line.startswith("#"):
            continue
        if ":" in line and not line.startswith("<?php") and not line.startswith("{%") and not line.startswith("{{"):
            yaml_like += 1
    return yaml_like >= max(1, len(sample) // 3)


def _detect_autofence_language(output: str, prompt: dict[str, Any]) -> str:
    instruction = str(prompt.get("instruction", "")).lower()
    category = str(prompt.get("category", "")).lower()
    requires_php = bool(prompt.get("requires_php", False))
    output_lstrip = output.lstrip()
    if output_lstrip.startswith("<?php") or requires_php:
        return "php"
    if "{{" in output or "{%" in output or category == "twig":
        return "twig"
    if "yaml" in instruction or category in {"routing", "di"}:
        if _looks_like_yaml(output):
            return "yaml"
    if _looks_like_yaml(output):
        return "yaml"
    return ""


def _normalize_output_with_interim_autofencing(
    output: str,
    *,
    prompt: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> tuple[str, list[str]]:
    interim_cfg = dict(eval_cfg.get("interim_autofencing", {}))
    if not bool(interim_cfg.get("enabled", False)):
        return output, []
    if not bool(interim_cfg.get("apply_before_contract_checks", True)):
        return output, []

    normalized = output.strip()
    if not normalized:
        return normalized, []
    if _extract_fenced_code_blocks(normalized):
        return normalized, []

    flags: list[str] = []
    if "```" in normalized:
        normalized = normalized.replace("```", "").strip()
        flags.append("removed_orphan_backticks")

    language = _detect_autofence_language(normalized, prompt)
    if not language:
        return normalized, flags

    wrapped = f"```{language}\n{normalized}\n```"
    flags.append(f"wrapped_{language}_fence")
    return wrapped, flags


def _prompt_expected_contract(prompt: dict[str, Any], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    instruction_lower = str(prompt.get("instruction", "")).strip().lower()
    expected_blocks = prompt.get("expected_fenced_blocks")
    if expected_blocks is None:
        if "two fenced blocks" in instruction_lower:
            expected_blocks = 2
        elif bool(prompt.get("require_fenced_php", False)) and bool(prompt.get("requires_php", False)):
            expected_blocks = 1
    expected_blocks_int = int(expected_blocks) if isinstance(expected_blocks, int) and expected_blocks > 0 else None

    expected_languages = [
        str(language).strip().lower()
        for language in prompt.get("expected_fenced_languages", [])
        if str(language).strip()
    ]
    if not expected_languages and expected_blocks_int == 1 and bool(prompt.get("requires_php", False)):
        expected_languages = ["php"]
    if not expected_languages and expected_blocks_int == 2 and "yaml" in instruction_lower:
        expected_languages = ["yaml", "php"]

    enforce_no_outside = prompt.get("enforce_no_outside_prose")
    if enforce_no_outside is None:
        enforce_no_outside = bool(eval_cfg.get("enforce_no_outside_prose_for_php_required", True))

    return {
        "expected_fenced_blocks": expected_blocks_int,
        "expected_fenced_languages": expected_languages,
        "enforce_no_outside_prose": bool(enforce_no_outside),
    }


def _has_explicit_fenced_php_block(output: str) -> bool:
    for block in _extract_fenced_code_blocks(output):
        if block["language"] in {"php", "phtml"}:
            return True
    return False


def _artifact_guard(output: str, generation_guardrails: dict[str, Any] | None = None) -> dict[str, Any]:
    has_wrapper = _has_prompt_wrapper_leakage(output)
    has_special = _has_special_token_artifact(output)
    guardrail_triggered = bool((generation_guardrails or {}).get("truncated_on_marker", False))
    reasons: list[str] = []
    if has_wrapper:
        reasons.append("prompt_wrapper_leakage")
    if has_special:
        reasons.append("special_token_or_fim_artifact")
    if guardrail_triggered:
        reasons.append("generation_stop_triggered_artifact")
    return {
        "is_clean": len(reasons) == 0,
        "has_prompt_wrapper_leakage": has_wrapper,
        "has_special_token_artifact": has_special or guardrail_triggered,
        "generation_stop_triggered": guardrail_triggered,
        "reasons": reasons,
    }


def _read_php_snippet_policy(config: dict[str, Any]) -> str:
    policy = str(config.get("php_snippet_policy", "php_only")).strip().lower()
    return policy if policy in {"php_only", "all_fences"} else "php_only"


def _select_snippets_for_checks(output: str, eval_cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    max_snippets = max(1, int(eval_cfg.get("max_code_checks_per_response", 3)))
    policy = _read_php_snippet_policy(eval_cfg)
    blocks = _extract_fenced_code_blocks(output)
    all_codes = [block["code"] for block in blocks]

    if policy == "all_fences":
        candidates = list(all_codes)
    else:
        candidates = []
        for block in blocks:
            language = block["language"]
            code = block["code"]
            if language in {"php", "phtml"}:
                candidates.append(code)
                continue
            if not language and "<?php" in code:
                candidates.append(code)

    if not candidates and "<?php" in output:
        candidates = [output.strip()]

    selected = candidates[:max_snippets]
    metadata = {
        "code_block_count": len(all_codes),
        "php_candidate_count": len(candidates),
        "php_checked_count": len(selected),
        "php_selection_policy": policy,
    }
    return selected, metadata


def _compute_format_sanity(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    numeric_streak = 0
    current_streak = 0
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
        if NUMERIC_LINE_RE.match(line):
            current_streak += 1
            if current_streak > numeric_streak:
                numeric_streak = current_streak
        else:
            current_streak = 0

    repeated_line_ratio = (max(counts.values()) / len(lines)) if len(lines) >= 20 and counts else 0.0
    has_prompt_wrapper_echo = _has_prompt_wrapper_leakage(output)
    has_special_token_artifact = _has_special_token_artifact(output)

    penalties = 0.0
    if has_prompt_wrapper_echo:
        penalties += 0.6
    if has_special_token_artifact:
        penalties += 0.7
    if numeric_streak >= 40:
        penalties += 0.4
    if repeated_line_ratio >= 0.25:
        penalties += 0.2

    score = max(0.0, round(1.0 - penalties, 4))
    return {
        "score": score,
        "has_prompt_wrapper_echo": has_prompt_wrapper_echo,
        "has_special_token_artifact": has_special_token_artifact,
        "numeric_line_streak": numeric_streak,
        "repeated_line_ratio": round(repeated_line_ratio, 4),
        "is_sane": score >= 0.8,
    }


def _required_checks_for_prompt(
    prompt_id: str,
    output: str,
    prompt: dict[str, Any] | None = None,
    eval_cfg: dict[str, Any] | None = None,
    contract_diagnostics: dict[str, Any] | None = None,
) -> tuple[dict[str, bool], list[str]]:
    prompt_cfg = prompt or {}
    eval_config = eval_cfg or {}
    contract_info = contract_diagnostics or _contract_diagnostics(output)
    checks: dict[str, bool] = {"non_empty_output": bool(output.strip())}
    lower = output.lower()

    if prompt_id == "block_attribute":
        checks.update(
            {
                "has_php_tag": "<?php" in output,
                "uses_block_attribute": "#[Block" in output,
                "contains_block_id": "gym_stats" in lower,
                "contains_block_label": "gym statistics" in lower,
            }
        )
        required = ["non_empty_output", "has_php_tag", "uses_block_attribute", "contains_block_id", "contains_block_label"]
    elif prompt_id == "service_di":
        checks.update(
            {
                "has_services_yaml": "services:" in lower,
                "references_logger_factory": "logger.factory" in lower,
                "has_constructor": "__construct(" in output,
                "uses_injected_logger": "logger" in lower,
            }
        )
        required = ["non_empty_output", "has_services_yaml", "references_logger_factory", "has_constructor"]
    elif prompt_id == "routing_yaml":
        checks.update(
            {
                "mentions_routing_yml": "routing.yml" in lower or ".routing.yml" in lower,
                "contains_path": "/gym/stats" in lower,
                "contains_controller": "_controller" in lower or "controller" in lower,
            }
        )
        required = ["non_empty_output", "mentions_routing_yml", "contains_path", "contains_controller"]
    elif prompt_id == "sdc_component":
        checks.update(
            {
                "mentions_component_yml": "component.yml" in lower,
                "mentions_components_directory": "components/" in lower or "/components" in lower,
                "mentions_template": ".twig" in lower or "template" in lower,
            }
        )
        required = ["non_empty_output", "mentions_component_yml", "mentions_components_directory"]
    else:
        required = ["non_empty_output"]

    if bool(prompt_cfg.get("require_fenced_php", False)):
        checks["has_fenced_php_block"] = _has_explicit_fenced_php_block(output)
        required.append("has_fenced_php_block")

    required_substrings = [str(value).strip() for value in prompt_cfg.get("required_substrings", []) if str(value).strip()]
    for term in required_substrings:
        key = f"contains::{term.lower()}"
        checks[key] = term.lower() in lower
        required.append(key)

    required_regex = [str(value).strip() for value in prompt_cfg.get("required_regex", []) if str(value).strip()]
    for pattern in required_regex:
        key = f"regex::{pattern}"
        try:
            checks[key] = bool(re.search(pattern, output, re.IGNORECASE | re.MULTILINE))
        except re.error:
            checks[key] = False
        required.append(key)

    strict_contract_mode = bool(eval_config.get("strict_contract_mode", True))
    requires_php = bool(prompt_cfg.get("requires_php", prompt_id in PROMPTS_REQUIRING_PHP_SNIPPET))
    if strict_contract_mode and requires_php and bool(prompt_cfg):
        contract_expectations = _prompt_expected_contract(prompt_cfg, eval_config)
        expected_blocks = contract_expectations["expected_fenced_blocks"]
        expected_languages = contract_expectations["expected_fenced_languages"]

        if expected_blocks is not None:
            checks["fenced_block_count"] = int(contract_info.get("fenced_block_count", 0)) == expected_blocks
            required.append("fenced_block_count")

        if expected_languages:
            actual_languages = [str(value).strip().lower() for value in contract_info.get("fenced_block_languages", [])]
            checks["fenced_language_order"] = actual_languages == expected_languages
            required.append("fenced_language_order")

        if bool(contract_expectations["enforce_no_outside_prose"]):
            checks["no_outside_prose"] = not bool(contract_info.get("outside_prose_detected", False))
            required.append("no_outside_prose")

    # Keep required order stable without duplicates.
    seen: set[str] = set()
    required = [name for name in required if not (name in seen or seen.add(name))]
    return checks, required


def _apply_external_required_checks(
    prompt_id: str,
    prompt: dict[str, Any] | None,
    checks: dict[str, bool],
    required: list[str],
    external_checks: dict[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    updated_checks = dict(checks)
    updated_required = list(required)
    requires_php = bool((prompt or {}).get("requires_php", prompt_id in PROMPTS_REQUIRING_PHP_SNIPPET))
    if requires_php:
        has_php_snippet = int(external_checks.get("php_checked_count", 0)) > 0
        updated_checks["has_php_snippet"] = has_php_snippet
        if "has_php_snippet" not in updated_required:
            updated_required.append("has_php_snippet")
    return updated_checks, updated_required


def _write_temp_php(snippet: str) -> Path:
    content = snippet.strip()
    if not content.lstrip().startswith("<?php"):
        content = "<?php\n" + content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as handle:
        handle.write(content)
        return Path(handle.name)


def _run_php_lint(snippets: list[str]) -> dict[str, Any]:
    php_bin = shutil.which("php")
    summary: dict[str, Any] = {
        "enabled": True,
        "available": bool(php_bin),
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "errors": [],
    }
    if not php_bin or not snippets:
        return summary

    for index, snippet in enumerate(snippets, start=1):
        tmp_path = _write_temp_php(snippet)
        try:
            proc = subprocess.run(
                [php_bin, "-l", str(tmp_path)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        summary["checked"] += 1
        if proc.returncode == 0:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
            message = (proc.stderr or proc.stdout or "").strip()
            summary["errors"].append({"snippet": index, "message": message[:500]})
    return summary


def _has_drupal_phpcs_standard(phpcs_bin: str) -> bool:
    proc = subprocess.run(
        [phpcs_bin, "-i"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "Drupal" in output


def _phpcs_runtime_misconfigured(output: str) -> bool:
    lower = output.lower()
    return (
        "referenced sniff" in lower and "does not exist" in lower
    ) or "coding standard \"drupal\" is not installed" in lower


def _parse_phpcs_report(stdout: str) -> dict[str, Any] | None:
    payload = (stdout or "").strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_tempfile_phpcs_noise(message: dict[str, Any]) -> bool:
    source = str(message.get("source", "")).strip()
    text = str(message.get("message", "")).strip()
    return source in PHPCS_TEMPFILE_NOISE_SOURCES or bool(PHPCS_TEMPFILE_CLASSNAME_RE.search(text))


def _collect_phpcs_messages(report: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for file_info in report.get("files", {}).values():
        if not isinstance(file_info, dict):
            continue
        for item in file_info.get("messages", []):
            if isinstance(item, dict):
                messages.append(item)
    return messages


def _format_phpcs_messages(messages: list[dict[str, Any]], max_items: int = 5) -> str:
    lines: list[str] = []
    for item in messages[:max_items]:
        line = item.get("line")
        source = item.get("source")
        text = str(item.get("message", "")).strip()
        prefix = f"line {line}" if line is not None else "line ?"
        if source:
            prefix += f" [{source}]"
        lines.append(f"{prefix}: {text}")
    if len(messages) > max_items:
        lines.append(f"... and {len(messages) - max_items} more issue(s)")
    return "; ".join(lines)


def _run_phpcs(snippets: list[str]) -> dict[str, Any]:
    phpcs_bin = shutil.which("phpcs")
    summary: dict[str, Any] = {
        "enabled": True,
        "available": bool(phpcs_bin),
        "drupal_standard_available": False,
        "runtime_broken": False,
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "errors": [],
    }
    if not phpcs_bin:
        return summary

    summary["drupal_standard_available"] = _has_drupal_phpcs_standard(phpcs_bin)
    if not summary["drupal_standard_available"] or not snippets:
        return summary

    for index, snippet in enumerate(snippets, start=1):
        tmp_path = _write_temp_php(snippet)
        try:
            proc = subprocess.run(
                [phpcs_bin, "--standard=Drupal", "--report=json", str(tmp_path)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        summary["checked"] += 1
        if proc.returncode == 0:
            summary["passed"] += 1
        else:
            combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if _phpcs_runtime_misconfigured(combined):
                summary["runtime_broken"] = True
                summary["drupal_standard_available"] = False
                summary["checked"] = 0
                summary["passed"] = 0
                summary["failed"] = 0
                summary["errors"] = [{"snippet": index, "message": combined[:500]}]
                break
            report = _parse_phpcs_report(proc.stdout or "")
            if report is not None:
                all_messages = _collect_phpcs_messages(report)
                relevant = [item for item in all_messages if not _is_tempfile_phpcs_noise(item)]
                if not relevant:
                    summary["passed"] += 1
                    continue
                summary["failed"] += 1
                message = _format_phpcs_messages(relevant)
                summary["errors"].append({"snippet": index, "message": message[:500]})
                continue

            summary["failed"] += 1
            summary["errors"].append({"snippet": index, "message": combined[:500]})
    return summary


def _extract_phpstan_messages(stdout: str, stderr: str) -> list[str]:
    messages: list[str] = []
    payload = (stdout or "").strip()
    if payload:
        try:
            data = json.loads(payload)
            for value in data.get("errors", []):
                text = str(value).strip()
                if text:
                    messages.append(text)
            for file_info in data.get("files", {}).values():
                for item in file_info.get("messages", []):
                    text = str(item.get("message", "")).strip()
                    if text:
                        messages.append(text)
        except json.JSONDecodeError:
            for line in payload.splitlines():
                text = line.strip()
                if text:
                    messages.append(text)
    if not messages:
        for line in (stderr or "").splitlines():
            text = line.strip()
            if text:
                messages.append(text)
    return messages


def _run_phpstan(snippets: list[str], failure_mode: str = "syntax_only") -> dict[str, Any]:
    phpstan_bin = shutil.which("phpstan")
    mode = str(failure_mode).strip().lower()
    if mode not in {"syntax_only", "strict"}:
        mode = "syntax_only"
    summary: dict[str, Any] = {
        "enabled": True,
        "available": bool(phpstan_bin),
        "failure_mode": mode,
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "syntax_errors": 0,
        "errors": [],
    }
    if not phpstan_bin or not snippets:
        return summary

    for index, snippet in enumerate(snippets, start=1):
        tmp_path = _write_temp_php(snippet)
        try:
            proc = subprocess.run(
                [
                    phpstan_bin,
                    "analyse",
                    "--no-progress",
                    "--error-format=json",
                    "--level=0",
                    str(tmp_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        summary["checked"] += 1
        messages = _extract_phpstan_messages(proc.stdout, proc.stderr)
        syntax_hits = [message for message in messages if PHPSTAN_SYNTAX_ERROR_RE.search(message)]
        summary["syntax_errors"] += len(syntax_hits)
        if mode == "syntax_only":
            failed = len(syntax_hits) > 0
        else:
            failed = proc.returncode != 0

        if failed:
            summary["failed"] += 1
            message = "; ".join(syntax_hits if mode == "syntax_only" and syntax_hits else messages)[:500]
            summary["errors"].append({"snippet": index, "message": message or "phpstan analysis failed"})
        else:
            summary["passed"] += 1
    return summary


def _run_external_checks(output: str, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    snippets, snippet_meta = _select_snippets_for_checks(output, eval_cfg)

    external = {
        "code_block_count": int(snippet_meta.get("code_block_count", 0)),
        "php_candidate_count": int(snippet_meta.get("php_candidate_count", 0)),
        "php_checked_count": int(snippet_meta.get("php_checked_count", 0)),
        "php_selection_policy": str(snippet_meta.get("php_selection_policy", _read_php_snippet_policy(eval_cfg))),
        "php_lint": {
            "enabled": bool(eval_cfg.get("run_php_lint", True)),
            "available": False,
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        },
        "phpcs": {
            "enabled": bool(eval_cfg.get("run_phpcs", False)),
            "available": False,
            "drupal_standard_available": False,
            "runtime_broken": False,
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        },
        "phpstan": {
            "enabled": bool(eval_cfg.get("run_phpstan", False)),
            "available": False,
            "failure_mode": str(eval_cfg.get("phpstan_failure_mode", "syntax_only")).strip().lower(),
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "syntax_errors": 0,
            "errors": [],
        },
    }

    if eval_cfg.get("run_php_lint", True):
        external["php_lint"] = _run_php_lint(snippets)

    if eval_cfg.get("run_phpcs", False):
        external["phpcs"] = _run_phpcs(snippets)
    if eval_cfg.get("run_phpstan", False):
        external["phpstan"] = _run_phpstan(
            snippets,
            failure_mode=str(eval_cfg.get("phpstan_failure_mode", "syntax_only")),
        )

    return external


def _score_result(
    required_checks: dict[str, bool],
    required: list[str],
    external_checks: dict[str, Any],
    artifact_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_total = len(required)
    required_passed = sum(1 for name in required if required_checks.get(name))
    required_score = (required_passed / required_total) if required_total else 1.0
    requires_php_snippet = "has_php_snippet" in required
    php_checked_count = int(external_checks.get("php_checked_count", 0))
    missing_required_php_snippet = requires_php_snippet and php_checked_count == 0

    semantic_weight = 1.0
    semantic_score = required_score
    style_weight = 0.0
    style_score = 1.0

    php_lint = external_checks.get("php_lint", {})
    if php_lint.get("enabled") and php_lint.get("available") and php_lint.get("checked", 0) > 0:
        semantic_weight += 0.2
        semantic_score += 0.2 * (php_lint.get("passed", 0) / max(1, php_lint.get("checked", 0)))
    elif missing_required_php_snippet and php_lint.get("enabled") and php_lint.get("available"):
        semantic_weight += 0.2

    phpstan = external_checks.get("phpstan", {})
    if phpstan.get("enabled") and phpstan.get("available") and phpstan.get("checked", 0) > 0:
        semantic_weight += 0.2
        semantic_score += 0.2 * (phpstan.get("passed", 0) / max(1, phpstan.get("checked", 0)))
    elif missing_required_php_snippet and phpstan.get("enabled") and phpstan.get("available"):
        semantic_weight += 0.2

    phpcs = external_checks.get("phpcs", {})
    if (
        phpcs.get("enabled")
        and phpcs.get("available")
        and phpcs.get("drupal_standard_available")
        and phpcs.get("checked", 0) > 0
    ):
        style_weight += 1.0
        style_score = phpcs.get("passed", 0) / max(1, phpcs.get("checked", 0))
    elif (
        missing_required_php_snippet
        and phpcs.get("enabled")
        and phpcs.get("available")
        and phpcs.get("drupal_standard_available")
    ):
        style_weight += 1.0
        style_score = 0.0

    semantic_score = semantic_score / max(semantic_weight, 1.0)
    semantic_score = round(max(0.0, min(1.0, semantic_score)), 4)
    style_score = round(max(0.0, min(1.0, style_score if style_weight > 0 else 1.0)), 4)

    guard = artifact_guard or {"is_clean": True}
    passes_artifact_guard = bool(guard.get("is_clean", True))
    if not passes_artifact_guard:
        semantic_score = 0.0
        style_score = min(style_score, 0.2)

    passes_required = required_passed == required_total
    passes_php_lint = not (
        (php_lint.get("enabled") and php_lint.get("available") and php_lint.get("checked", 0) > 0 and php_lint.get("failed", 0) > 0)
        or (missing_required_php_snippet and php_lint.get("enabled") and php_lint.get("available"))
    )
    passes_phpcs = not (
        (
            phpcs.get("enabled")
            and phpcs.get("available")
            and phpcs.get("drupal_standard_available")
            and phpcs.get("checked", 0) > 0
            and phpcs.get("failed", 0) > 0
        )
        or (
            missing_required_php_snippet
            and phpcs.get("enabled")
            and phpcs.get("available")
            and phpcs.get("drupal_standard_available")
        )
    )
    passes_phpstan = not (
        (
            phpstan.get("enabled")
            and phpstan.get("available")
            and phpstan.get("checked", 0) > 0
            and phpstan.get("failed", 0) > 0
        )
        or (missing_required_php_snippet and phpstan.get("enabled") and phpstan.get("available"))
    )

    passes_semantic = passes_required and passes_php_lint and passes_phpstan and passes_artifact_guard
    passes_style = passes_phpcs
    overall_score = round((semantic_score * 0.8) + (style_score * 0.2), 4)

    return {
        "semantic_score": semantic_score,
        "style_score": style_score,
        "overall_score": overall_score,
        "score": overall_score,
        "required_total": required_total,
        "required_passed": required_passed,
        "required_score": round(required_score, 4),
        "passes_required": passes_required,
        "passes_php_lint": passes_php_lint,
        "passes_phpcs": passes_phpcs,
        "passes_phpstan": passes_phpstan,
        "passes_artifact_guard": passes_artifact_guard,
        "passes_semantic": passes_semantic,
        "passes_style": passes_style,
        "passed": passes_semantic,
        "style_passed": passes_style,
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    model_names = sorted(set(result["model_name"] for result in results))
    per_model: list[dict[str, Any]] = []
    all_fine_scores: list[float] = []
    all_base_scores: list[float] = []
    all_fine_semantic_scores: list[float] = []
    all_base_semantic_scores: list[float] = []
    all_fine_style_scores: list[float] = []
    all_base_style_scores: list[float] = []

    for model_name in model_names:
        model_results = [result for result in results if result["model_name"] == model_name]
        fine = [result for result in model_results if result["variant"] == "fine_tuned"]
        base = [result for result in model_results if result["variant"] == "baseline"]

        fine_scores = [float(item["score"]) for item in fine]
        base_scores = [float(item["score"]) for item in base]
        fine_semantic_scores = [float(item.get("semantic_score", item["score"])) for item in fine]
        base_semantic_scores = [float(item.get("semantic_score", item["score"])) for item in base]
        fine_style_scores = [float(item.get("style_score", 1.0)) for item in fine]
        base_style_scores = [float(item.get("style_score", 1.0)) for item in base]
        fine_format_scores = [float(item.get("format_sanity", {}).get("score", 1.0)) for item in fine]
        base_format_scores = [float(item.get("format_sanity", {}).get("score", 1.0)) for item in base]
        all_fine_scores.extend(fine_scores)
        all_base_scores.extend(base_scores)
        all_fine_semantic_scores.extend(fine_semantic_scores)
        all_base_semantic_scores.extend(base_semantic_scores)
        all_fine_style_scores.extend(fine_style_scores)
        all_base_style_scores.extend(base_style_scores)

        by_prompt = {item["prompt_id"]: item for item in base}
        prompt_deltas = []
        fine_wins = 0
        baseline_wins = 0
        ties = 0
        for fine_item in fine:
            baseline_item = by_prompt.get(fine_item["prompt_id"])
            if not baseline_item:
                continue
            delta = round(float(fine_item["score"]) - float(baseline_item["score"]), 4)
            semantic_delta = round(
                float(fine_item.get("semantic_score", fine_item["score"]))
                - float(baseline_item.get("semantic_score", baseline_item["score"])),
                4,
            )
            style_delta = round(float(fine_item.get("style_score", 1.0)) - float(baseline_item.get("style_score", 1.0)), 4)
            if delta > 0:
                fine_wins += 1
            elif delta < 0:
                baseline_wins += 1
            else:
                ties += 1
            prompt_deltas.append(
                {
                    "prompt_id": fine_item["prompt_id"],
                    "category": fine_item["category"],
                    "fine_tuned_score": fine_item["score"],
                    "baseline_score": baseline_item["score"],
                    "delta": delta,
                    "semantic_delta": semantic_delta,
                    "style_delta": style_delta,
                }
            )

        fine_avg = _average(fine_scores)
        base_avg = _average(base_scores)
        fine_semantic_avg = _average(fine_semantic_scores)
        base_semantic_avg = _average(base_semantic_scores)
        fine_style_avg = _average(fine_style_scores)
        base_style_avg = _average(base_style_scores)
        per_model.append(
            {
                "model_name": model_name,
                "base_model": fine[0].get("base_model") if fine else (base[0].get("base_model") if base else ""),
                "fine_tuned_avg_score": fine_avg,
                "baseline_avg_score": base_avg,
                "delta_avg_score": round(fine_avg - base_avg, 4),
                "fine_tuned_avg_semantic_score": fine_semantic_avg,
                "baseline_avg_semantic_score": base_semantic_avg,
                "delta_avg_semantic_score": round(fine_semantic_avg - base_semantic_avg, 4),
                "fine_tuned_avg_style_score": fine_style_avg,
                "baseline_avg_style_score": base_style_avg,
                "delta_avg_style_score": round(fine_style_avg - base_style_avg, 4),
                "fine_tuned_pass_rate": _average([1.0 if item["passed"] else 0.0 for item in fine]),
                "baseline_pass_rate": _average([1.0 if item["passed"] else 0.0 for item in base]),
                "fine_tuned_style_pass_rate": _average([1.0 if item.get("passes_style", True) else 0.0 for item in fine]),
                "baseline_style_pass_rate": _average([1.0 if item.get("passes_style", True) else 0.0 for item in base]),
                "fine_tuned_format_sanity_avg": _average(fine_format_scores),
                "baseline_format_sanity_avg": _average(base_format_scores),
                "delta_format_sanity_avg": round(_average(fine_format_scores) - _average(base_format_scores), 4),
                "fine_tuned_format_sanity_fail_rate": _average(
                    [1.0 if not item.get("format_sanity", {}).get("is_sane", True) else 0.0 for item in fine]
                ),
                "baseline_format_sanity_fail_rate": _average(
                    [1.0 if not item.get("format_sanity", {}).get("is_sane", True) else 0.0 for item in base]
                ),
                "fine_tuned_wins": fine_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "prompt_deltas": prompt_deltas,
            }
        )

    return {
        "model_count": len(model_names),
        "overall_fine_tuned_avg_score": _average(all_fine_scores),
        "overall_baseline_avg_score": _average(all_base_scores),
        "overall_delta_avg_score": round(_average(all_fine_scores) - _average(all_base_scores), 4),
        "overall_fine_tuned_avg_semantic_score": _average(all_fine_semantic_scores),
        "overall_baseline_avg_semantic_score": _average(all_base_semantic_scores),
        "overall_delta_avg_semantic_score": round(
            _average(all_fine_semantic_scores) - _average(all_base_semantic_scores), 4
        ),
        "overall_fine_tuned_avg_style_score": _average(all_fine_style_scores),
        "overall_baseline_avg_style_score": _average(all_base_style_scores),
        "overall_delta_avg_style_score": round(_average(all_fine_style_scores) - _average(all_base_style_scores), 4),
        "models": per_model,
    }


def _checkpoint_sweep_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("variant") != "fine_tuned":
            continue
        model_name = str(result.get("model_name", ""))
        by_model.setdefault(model_name, []).append(result)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for model_name, rows in by_model.items():
        root_name = model_name.split("@", 1)[0]
        semantic_pass_rate = _average([1.0 if bool(row.get("passes_semantic", False)) else 0.0 for row in rows])
        required_pass_rate = _average([1.0 if bool(row.get("passes_required", False)) else 0.0 for row in rows])
        avg_score = _average([float(row.get("score", 0.0)) for row in rows])
        grouped.setdefault(root_name, []).append(
            {
                "model_name": model_name,
                "semantic_pass_rate": semantic_pass_rate,
                "required_pass_rate": required_pass_rate,
                "avg_score": avg_score,
            }
        )

    summary: list[dict[str, Any]] = []
    for root_name, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["semantic_pass_rate"],
                item["required_pass_rate"],
                item["avg_score"],
            ),
            reverse=True,
        )
        summary.append(
            {
                "base_model_name": root_name,
                "selected_model_name": ordered[0]["model_name"],
                "selection_rule": [
                    "semantic_pass_rate",
                    "required_pass_rate",
                    "avg_score",
                ],
                "candidates": ordered,
            }
        )
    return summary


def _hash_directory(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        return ""
    digest = __import__("hashlib").sha256()
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(calculate_hash(file_path).encode("utf-8"))
    return digest.hexdigest()


def _write_sample_index(sample_outputs_dir: Path) -> Path:
    index_path = sample_outputs_dir / "index.json"
    entries = []
    for file_path in sorted(sample_outputs_dir.rglob("*.txt")):
        entries.append(
            {
                "path": file_path.relative_to(sample_outputs_dir.parent).as_posix(),
                "sha256": calculate_hash(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    return index_path


def _adapter_subdir_for_mode(mode: str) -> str:
    return "final" if mode in {"full_scale", "final"} else "test_run"


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


def _load_tokenizer_for_evaluation(*, model_id: str, logger: PipelineLogger, auto_tokenizer_cls):
    try:
        tokenizer = auto_tokenizer_cls.from_pretrained(model_id, trust_remote_code=True)
    except ValueError as exc:
        error_text = str(exc)
        if "Tokenizer class TokenizersBackend does not exist or is not currently imported." not in error_text:
            raise
        try:
            from transformers import MistralCommonTokenizer
        except Exception as import_exc:
            raise RuntimeError(
                "Failed to load tokenizer via AutoTokenizer due to a TokenizersBackend class mismatch. "
                "Install or upgrade `mistral-common` and `transformers`, then retry stage 8."
            ) from import_exc
        logger.info(
            "AutoTokenizer failed with TokenizersBackend mismatch; using MistralCommonTokenizer fallback.",
            model=model_id,
        )
        tokenizer = MistralCommonTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def _checkpoint_targets(model_root: Path, sweep_cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    if not bool(sweep_cfg.get("enabled", False)):
        return []

    checkpoints = []
    for path in sorted(model_root.glob("checkpoint-*"), key=_checkpoint_step):
        if not path.is_dir():
            continue
        if not (path / "adapter_model.safetensors").exists():
            continue
        step = _checkpoint_step(path)
        if step < 0:
            continue
        checkpoints.append((step, path))

    explicit_steps = {
        int(step)
        for step in sweep_cfg.get("explicit_steps", [])
        if str(step).strip().isdigit()
    }
    step_interval = int(sweep_cfg.get("step_interval", 0))
    selected = []
    for step, path in checkpoints:
        if explicit_steps and step in explicit_steps:
            selected.append((f"checkpoint-{step}", path))
            continue
        if step_interval > 0 and step % step_interval == 0:
            selected.append((f"checkpoint-{step}", path))

    max_checkpoints = int(sweep_cfg.get("max_checkpoints", 0))
    if max_checkpoints > 0 and len(selected) > max_checkpoints:
        selected = selected[-max_checkpoints:]
    return selected


def _reset_sample_outputs_dir(sample_outputs_dir: Path) -> None:
    if sample_outputs_dir.exists():
        shutil.rmtree(sample_outputs_dir)
    sample_outputs_dir.mkdir(parents=True, exist_ok=True)


def _load_model_for_evaluation(
    *,
    model_name: str,
    base_model_id: str,
    adapter_path: Path,
    eval_cfg: dict[str, Any],
    eval_dir: Path,
    logger: PipelineLogger,
    torch_module,
    auto_tokenizer_cls,
    auto_model_cls,
    peft_model_cls,
):
    tokenizer = _load_tokenizer_for_evaluation(
        model_id=base_model_id,
        logger=logger,
        auto_tokenizer_cls=auto_tokenizer_cls,
    )
    is_ministral3_model = "ministral-3" in base_model_id.lower() or "ministral-3" in model_name.lower()
    model_loader_cls = auto_model_cls
    model_config_obj = None
    if is_ministral3_model:
        from transformers import AutoConfig, Mistral3ForConditionalGeneration

        model_loader_cls = Mistral3ForConditionalGeneration
        model_config_obj = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
        model_config_obj = _strip_pretrained_quantization_config(model_config_obj, logger)
        model_config_obj = _coerce_model_config_object(model_config_obj, logger)

    requested_device = str(eval_cfg.get("device", "auto")).lower()
    cuda_available = torch_module.cuda.is_available()
    offload_dir = eval_dir / "offload" / _sanitize_slug(model_name)
    offload_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    if requested_device == "cpu":
        attempts = [{"label": "cpu", "device_map": "cpu", "torch_dtype": torch_module.float32}]
    elif requested_device == "cuda":
        if cuda_available:
            attempts.append({"label": "cuda_auto", "device_map": "auto", "torch_dtype": torch_module.float16})
        attempts.append({"label": "cpu_fallback", "device_map": "cpu", "torch_dtype": torch_module.float32})
    else:
        if cuda_available:
            attempts.append({"label": "auto_cuda", "device_map": "auto", "torch_dtype": torch_module.float16})
        # On CPU-only hosts, loading directly on CPU is more reliable than device_map=auto.
        if cuda_available:
            attempts.append({"label": "auto_cpu", "device_map": "auto", "torch_dtype": torch_module.float32})
        attempts.append({"label": "cpu_fallback", "device_map": "cpu", "torch_dtype": torch_module.float32})

    last_error: Exception | None = None
    for attempt in attempts:
        load_kwargs = {
            "torch_dtype": attempt["torch_dtype"],
            "device_map": attempt["device_map"],
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if attempt["device_map"] == "auto":
            load_kwargs["offload_folder"] = str(offload_dir)
            load_kwargs["offload_state_dict"] = True

        try:
            logger.info(
                "Loading base model for evaluation.",
                model=model_name,
                attempt=attempt["label"],
                device_map=attempt["device_map"],
            )
            if is_ministral3_model:
                base_model = model_loader_cls.from_pretrained(
                    base_model_id,
                    config=model_config_obj,
                    attn_implementation="eager",
                    **load_kwargs,
                )
            else:
                base_model = auto_model_cls.from_pretrained(base_model_id, **load_kwargs)
            model = peft_model_cls.from_pretrained(base_model, str(adapter_path))
            model.eval()
            return tokenizer, base_model, model
        except Exception as exc:
            last_error = exc
            logger.info(
                "Model load attempt failed; trying next strategy.",
                model=model_name,
                attempt=attempt["label"],
                error=str(exc),
            )

    raise RuntimeError(str(last_error) if last_error else "Unable to load model for evaluation.")


def run_evaluation_stage(config: dict, logger: PipelineLogger, root: Path) -> int:
    eval_cfg = _read_eval_config(config)
    generation_profile = _build_generation_profile(eval_cfg)
    evaluator_metadata = _build_evaluator_metadata(eval_cfg, generation_profile=generation_profile)
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_outputs_dir = eval_dir / "sample_outputs"
    _reset_sample_outputs_dir(sample_outputs_dir)

    manifest = Manifest("evaluation", eval_dir)
    manifest.data["config"] = eval_cfg
    manifest.data["evaluator"] = evaluator_metadata
    manifest.data["generation_profile"] = generation_profile

    models_to_eval = _resolve_models_for_eval(config, eval_cfg)
    if not models_to_eval:
        logger.error("No models defined for evaluation.")
        return 1

    max_models = max(1, int(eval_cfg.get("max_models", 1)))
    models_to_eval = models_to_eval[:max_models]

    adapter_subdir = _adapter_subdir_for_mode(str(eval_cfg.get("mode", "test_run")))
    checkpoint_sweep_cfg = dict(eval_cfg.get("checkpoint_sweep", {}))
    ready_models: list[dict[str, Any]] = []
    blocked_models: list[dict[str, str]] = []

    for model_cfg in models_to_eval:
        model_name = str(model_cfg.get("name", "")).strip()
        base_model = str(model_cfg.get("base_model", "")).strip()
        if not model_name or not base_model:
            blocked_models.append(
                {
                    "model": model_name or "unknown",
                    "reason": "missing model name/base_model in config",
                }
            )
            continue

        model_root = root / "models" / model_name / adapter_subdir
        if not model_root.exists():
            blocked_models.append(
                {
                    "model": model_name,
                    "reason": f"model directory not found at {model_root}",
                }
            )
            continue

        targets: list[tuple[str, Path]] = []
        if bool(checkpoint_sweep_cfg.get("enabled", False)):
            targets.extend(_checkpoint_targets(model_root, checkpoint_sweep_cfg))
            if bool(checkpoint_sweep_cfg.get("include_adapter", True)):
                adapter_path = model_root / "adapter"
                if adapter_path.exists():
                    targets.append(("adapter", adapter_path))
        else:
            adapter_path = model_root / "adapter"
            if adapter_path.exists():
                targets.append(("adapter", adapter_path))

        if not targets:
            blocked_models.append(
                {
                    "model": model_name,
                    "reason": f"no evaluation targets found under {model_root}",
                }
            )
            continue

        seen_target_names: set[str] = set()
        for target_name, target_path in targets:
            if target_name in seen_target_names:
                continue
            seen_target_names.add(target_name)
            eval_model_name = model_name if target_name == "adapter" else f"{model_name}@{target_name}"
            manifest.add_input(
                f"adapter_{eval_model_name}",
                adapter_subdir,
                _hash_directory(target_path),
            )
            ready_models.append(
                {
                    "name": eval_model_name,
                    "source_model_name": model_name,
                    "base_model": base_model,
                    "adapter_path": target_path,
                    "target_name": target_name,
                    "prompt_template": _resolve_prompt_template(model_cfg),
                }
            )

    if not ready_models:
        logger.error("No adapters available for Stage 8 evaluation.", blocked_models=blocked_models)
        return 1

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as exc:
        logger.error(f"Evaluation dependencies are unavailable: {exc}")
        return 1

    seed = int(eval_cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_suite = eval_cfg["prompt_suite"]
    all_results: list[dict[str, Any]] = []

    for model_cfg in ready_models:
        model_name = model_cfg["name"]
        base_model_id = model_cfg["base_model"]
        adapter_path = model_cfg["adapter_path"]
        prompt_template = str(model_cfg.get("prompt_template", PROMPT_TEMPLATE_PLAIN))
        logger.info(f"Evaluating model {model_name} with adapter {adapter_path}")

        try:
            tokenizer, base_model, model = _load_model_for_evaluation(
                model_name=model_name,
                base_model_id=base_model_id,
                adapter_path=adapter_path,
                eval_cfg=eval_cfg,
                eval_dir=eval_dir,
                logger=logger,
                torch_module=torch,
                auto_tokenizer_cls=AutoTokenizer,
                auto_model_cls=AutoModelForCausalLM,
                peft_model_cls=PeftModel,
            )
        except Exception as exc:
            logger.error(f"Failed to load model {model_name}: {exc}")
            blocked_models.append({"model": model_name, "reason": f"model_load_failed: {exc}"})
            continue

        model_dir = sample_outputs_dir / _sanitize_slug(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        runtime_eval_cfg = dict(eval_cfg)
        runtime_eval_cfg["_bad_words_ids"] = _build_bad_words_ids(
            tokenizer, _string_list(eval_cfg.get("generation_blocklist_strings"), DEFAULT_GENERATION_BLOCKLIST_STRINGS)
        )
        runtime_eval_cfg["_compiled_stop_patterns"] = _compile_stop_patterns(
            _string_list(eval_cfg.get("generation_stop_regex"), DEFAULT_GENERATION_STOP_REGEX)
        )

        for variant in ["fine_tuned", "baseline"]:
            context = model.disable_adapter() if variant == "baseline" and hasattr(model, "disable_adapter") else nullcontext()
            with context:
                for prompt in prompt_suite:
                    prompt_id = str(prompt.get("id", "prompt"))
                    instruction = str(prompt.get("instruction", "")).strip()
                    input_text = str(prompt.get("input", ""))
                    category = str(prompt.get("category", "general"))

                    logger.info(f"Generating {variant} output for {model_name}:{prompt_id}")
                    generated_output = _generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        input_text=input_text,
                        max_new_tokens=int(runtime_eval_cfg["max_new_tokens"]),
                        eval_cfg=runtime_eval_cfg,
                        prompt_template=prompt_template,
                    )
                    raw_output, generation_guardrails = _truncate_on_generation_markers(
                        generated_output,
                        runtime_eval_cfg,
                    )
                    normalized_output, postprocess_flags = _normalize_output_with_interim_autofencing(
                        raw_output,
                        prompt=prompt,
                        eval_cfg=runtime_eval_cfg,
                    )

                    raw_contract_diagnostics = _contract_diagnostics(raw_output)
                    checks, required = _required_checks_for_prompt(
                        prompt_id,
                        raw_output,
                        prompt=prompt,
                        eval_cfg=runtime_eval_cfg,
                        contract_diagnostics=raw_contract_diagnostics,
                    )
                    external_checks = _run_external_checks(raw_output, runtime_eval_cfg)
                    checks, required = _apply_external_required_checks(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        checks=checks,
                        required=required,
                        external_checks=external_checks,
                    )
                    format_sanity = _compute_format_sanity(raw_output)
                    artifact_guard = _artifact_guard(raw_output, generation_guardrails=generation_guardrails)
                    score = _score_result(checks, required, external_checks, artifact_guard=artifact_guard)

                    normalized_contract_diagnostics = _contract_diagnostics(normalized_output)
                    normalized_checks, normalized_required = _required_checks_for_prompt(
                        prompt_id,
                        normalized_output,
                        prompt=prompt,
                        eval_cfg=runtime_eval_cfg,
                        contract_diagnostics=normalized_contract_diagnostics,
                    )
                    normalized_external_checks = external_checks
                    if normalized_output != raw_output:
                        normalized_external_checks = _run_external_checks(normalized_output, runtime_eval_cfg)
                    normalized_checks, normalized_required = _apply_external_required_checks(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        checks=normalized_checks,
                        required=normalized_required,
                        external_checks=normalized_external_checks,
                    )
                    normalized_format_sanity = _compute_format_sanity(normalized_output)
                    normalized_artifact_guard = _artifact_guard(
                        normalized_output,
                        generation_guardrails=generation_guardrails,
                    )
                    normalized_score = _score_result(
                        normalized_checks,
                        normalized_required,
                        normalized_external_checks,
                        artifact_guard=normalized_artifact_guard,
                    )

                    generated_output_path = model_dir / f"{variant}__{_sanitize_slug(prompt_id)}__generated.txt"
                    raw_output_path = model_dir / f"{variant}__{_sanitize_slug(prompt_id)}__raw.txt"
                    normalized_output_path = model_dir / f"{variant}__{_sanitize_slug(prompt_id)}__normalized.txt"
                    with open(generated_output_path, "w", encoding="utf-8") as handle:
                        handle.write(generated_output)
                    with open(raw_output_path, "w", encoding="utf-8") as handle:
                        handle.write(raw_output)
                    with open(normalized_output_path, "w", encoding="utf-8") as handle:
                        handle.write(normalized_output)

                    result = {
                        "timestamp": _iso_timestamp(),
                        "model_name": model_name,
                        "source_model_name": model_cfg.get("source_model_name", model_name),
                        "eval_target": model_cfg.get("target_name", "adapter"),
                        "base_model": base_model_id,
                        "prompt_template": prompt_template,
                        "variant": variant,
                        "prompt_id": prompt_id,
                        "category": category,
                        "instruction": instruction,
                        "input": input_text,
                        "output_file": raw_output_path.relative_to(root).as_posix(),
                        "generated_output_file": generated_output_path.relative_to(root).as_posix(),
                        "raw_output_file": raw_output_path.relative_to(root).as_posix(),
                        "normalized_output_file": normalized_output_path.relative_to(root).as_posix(),
                        "postprocess_flags": postprocess_flags,
                        "generated_output_length": len(generated_output),
                        "output_length": len(raw_output),
                        "normalized_output_length": len(normalized_output),
                        "checks": checks,
                        "required_checks": required,
                        "external_checks": external_checks,
                        "format_sanity": format_sanity,
                        "contract_diagnostics": raw_contract_diagnostics,
                        "generation_guardrails": generation_guardrails,
                        "artifact_guard": artifact_guard,
                        "normalized_checks": normalized_checks,
                        "normalized_required_checks": normalized_required,
                        "normalized_external_checks": normalized_external_checks,
                        "normalized_contract_diagnostics": normalized_contract_diagnostics,
                        "normalized_format_sanity": normalized_format_sanity,
                        "normalized_artifact_guard": normalized_artifact_guard,
                        "normalized_result": normalized_score,
                    }
                    result.update(score)
                    all_results.append(result)

        # Free as much memory as possible before next model.
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        logger.error("Evaluation produced no outputs.", blocked_models=blocked_models)
        return 1

    summary = summarize_results(all_results)
    checkpoint_sweep_summary = _checkpoint_sweep_summary(all_results)
    metrics = {
        "stage": "evaluation",
        "timestamp": _iso_timestamp(),
        "seed": seed,
        "mode": eval_cfg.get("mode", "test_run"),
        "prompt_suite": prompt_suite,
        "evaluator": evaluator_metadata,
        "generation_profile": generation_profile,
        "blocked_models": blocked_models,
        "summary": summary,
        "checkpoint_sweep_selection": checkpoint_sweep_summary,
        "results": all_results,
    }

    metrics_path = eval_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    sample_index_path = _write_sample_index(sample_outputs_dir)

    manifest.add_output("metrics", "eval/metrics.json", calculate_hash(metrics_path))
    manifest.add_output("sample_index", "eval/sample_outputs/index.json", calculate_hash(sample_index_path))
    manifest.set_metrics(
        {
            "models_evaluated": summary["model_count"],
            "overall_fine_tuned_avg_score": summary["overall_fine_tuned_avg_score"],
            "overall_baseline_avg_score": summary["overall_baseline_avg_score"],
            "overall_delta_avg_score": summary["overall_delta_avg_score"],
            "overall_fine_tuned_avg_semantic_score": summary["overall_fine_tuned_avg_semantic_score"],
            "overall_baseline_avg_semantic_score": summary["overall_baseline_avg_semantic_score"],
            "overall_delta_avg_semantic_score": summary["overall_delta_avg_semantic_score"],
            "overall_fine_tuned_avg_style_score": summary["overall_fine_tuned_avg_style_score"],
            "overall_baseline_avg_style_score": summary["overall_baseline_avg_style_score"],
            "overall_delta_avg_style_score": summary["overall_delta_avg_style_score"],
            "evaluator_logic_version": evaluator_metadata["logic_version"],
            "evaluator_logic_sha256": evaluator_metadata["logic_sha256"],
            "prompt_suite_sha256": evaluator_metadata["prompt_suite_sha256"],
            "generation_profile_sha256": generation_profile["generation_profile_sha256"],
            "artifact_blocklist_version": generation_profile["artifact_blocklist_version"],
            "blocked_models": len(blocked_models),
            "result_count": len(all_results),
            "checkpoint_sweep_selection_count": len(checkpoint_sweep_summary),
        }
    )
    manifest.save(eval_dir / "manifest.json")

    logger.info(
        "Stage 8 evaluation complete.",
        metrics_path=str(metrics_path),
        sample_outputs=str(sample_outputs_dir),
        models_evaluated=summary["model_count"],
    )
    return 0


__all__ = [
    "run_evaluation_stage",
    "summarize_results",
    "_build_generation_kwargs",
    "_contract_diagnostics",
    "_extract_code_blocks",
    "_normalize_output_with_interim_autofencing",
    "_required_checks_for_prompt",
    "_score_result",
    "_truncate_on_generation_markers",
]
