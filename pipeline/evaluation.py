import json
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash

DEFAULT_PROMPT_SUITE = [
    {
        "id": "block_attribute",
        "category": "attributes",
        "instruction": "Create a Drupal 11 Block plugin using PHP 8.3 attributes. The block ID should be 'gym_stats' and the label 'Gym Statistics'.",
        "input": "",
    },
    {
        "id": "service_di",
        "category": "di",
        "instruction": "Define a Drupal 11 service in gym.services.yml and its class implementation using constructor injection for the logger.factory service.",
        "input": "",
    },
    {
        "id": "routing_yaml",
        "category": "routing",
        "instruction": "Create a Drupal 11 gym.routing.yml route and a matching controller method for the path '/gym/stats'.",
        "input": "",
    },
    {
        "id": "sdc_component",
        "category": "sdc",
        "instruction": "Show a Drupal 11 Single Directory Component example with directory structure and component.yml.",
        "input": "",
    },
]
PROMPT_WRAPPER_RE = re.compile(r"(?mi)^\s*(instruction|input|output)\s*:")
NUMERIC_LINE_RE = re.compile(r"^\d{1,5}(?:[.):])?$")


def _iso_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "model"


def _read_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "seed": int(config.get("seed", 42)),
        "mode": "test_run",
        "max_new_tokens": 512,
        "device": "auto",
        "max_models": 1,
        "run_php_lint": True,
        "run_phpcs": False,
        "max_code_checks_per_response": 3,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "prompt_suite": DEFAULT_PROMPT_SUITE,
    }
    merged = defaults | config.get("evaluation", {})
    prompt_suite = merged.get("prompt_suite") or DEFAULT_PROMPT_SUITE
    merged["prompt_suite"] = prompt_suite
    merged["seed"] = int(merged.get("seed", defaults["seed"]))
    merged["max_new_tokens"] = int(merged.get("max_new_tokens", defaults["max_new_tokens"]))
    merged["max_models"] = int(merged.get("max_models", defaults["max_models"]))
    merged["max_code_checks_per_response"] = int(
        merged.get("max_code_checks_per_response", defaults["max_code_checks_per_response"])
    )
    merged["repetition_penalty"] = float(merged.get("repetition_penalty", defaults["repetition_penalty"]))
    merged["no_repeat_ngram_size"] = int(merged.get("no_repeat_ngram_size", defaults["no_repeat_ngram_size"]))
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


def _build_prompt(instruction: str, input_text: str = "") -> str:
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
    return kwargs


def _generate_response(model, tokenizer, instruction: str, input_text: str, max_new_tokens: int, eval_cfg: dict[str, Any]) -> str:
    prompt = _build_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = _model_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    with __import__("torch").no_grad():
        generation_kwargs = _build_generation_kwargs(tokenizer, max_new_tokens=max_new_tokens, eval_cfg=eval_cfg)
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_tokens = inputs["input_ids"].shape[1]
    response_tokens = outputs[0][prompt_tokens:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response.strip()


def _extract_code_blocks(output: str) -> list[str]:
    block_pattern = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", re.DOTALL)
    blocks = [match.group(1).strip() for match in block_pattern.finditer(output)]
    if blocks:
        return blocks
    if "<?php" in output:
        return [output.strip()]
    return []


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
    has_prompt_wrapper_echo = bool(PROMPT_WRAPPER_RE.search(output))

    penalties = 0.0
    if has_prompt_wrapper_echo:
        penalties += 0.6
    if numeric_streak >= 40:
        penalties += 0.4
    if repeated_line_ratio >= 0.25:
        penalties += 0.2

    score = max(0.0, round(1.0 - penalties, 4))
    return {
        "score": score,
        "has_prompt_wrapper_echo": has_prompt_wrapper_echo,
        "numeric_line_streak": numeric_streak,
        "repeated_line_ratio": round(repeated_line_ratio, 4),
        "is_sane": score >= 0.8,
    }


def _required_checks_for_prompt(prompt_id: str, output: str) -> tuple[dict[str, bool], list[str]]:
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

    return checks, required


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
    proc = subprocess.run([phpcs_bin, "-i"], check=False, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "Drupal" in output


def _run_phpcs(snippets: list[str]) -> dict[str, Any]:
    phpcs_bin = shutil.which("phpcs")
    summary: dict[str, Any] = {
        "enabled": True,
        "available": bool(phpcs_bin),
        "drupal_standard_available": False,
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
                [phpcs_bin, "--standard=Drupal", str(tmp_path)],
                check=False,
                capture_output=True,
                text=True,
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


def _run_external_checks(output: str, eval_cfg: dict[str, Any]) -> dict[str, Any]:
    snippets = _extract_code_blocks(output)
    max_snippets = max(1, int(eval_cfg.get("max_code_checks_per_response", 3)))
    snippets = snippets[:max_snippets]

    external = {
        "code_block_count": len(snippets),
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
            "checked": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
        },
    }

    if eval_cfg.get("run_php_lint", True):
        external["php_lint"] = _run_php_lint(snippets)

    if eval_cfg.get("run_phpcs", False):
        external["phpcs"] = _run_phpcs(snippets)

    return external


def _score_result(required_checks: dict[str, bool], required: list[str], external_checks: dict[str, Any]) -> dict[str, Any]:
    required_total = len(required)
    required_passed = sum(1 for name in required if required_checks.get(name))
    required_score = (required_passed / required_total) if required_total else 1.0

    lint_weight = 0.0
    lint_score = 1.0
    php_lint = external_checks.get("php_lint", {})
    if php_lint.get("enabled") and php_lint.get("available") and php_lint.get("checked", 0) > 0:
        lint_weight += 0.1
        lint_score *= php_lint.get("passed", 0) / max(1, php_lint.get("checked", 0))

    phpcs = external_checks.get("phpcs", {})
    if (
        phpcs.get("enabled")
        and phpcs.get("available")
        and phpcs.get("drupal_standard_available")
        and phpcs.get("checked", 0) > 0
    ):
        lint_weight += 0.1
        lint_score *= phpcs.get("passed", 0) / max(1, phpcs.get("checked", 0))

    base_weight = 1.0 - lint_weight
    score = (required_score * base_weight) + (lint_score * lint_weight)
    score = round(score, 4)

    passes_required = required_passed == required_total
    passes_php_lint = not (
        php_lint.get("enabled") and php_lint.get("available") and php_lint.get("checked", 0) > 0 and php_lint.get("failed", 0) > 0
    )
    passes_phpcs = not (
        phpcs.get("enabled")
        and phpcs.get("available")
        and phpcs.get("drupal_standard_available")
        and phpcs.get("checked", 0) > 0
        and phpcs.get("failed", 0) > 0
    )

    return {
        "score": score,
        "required_total": required_total,
        "required_passed": required_passed,
        "required_score": round(required_score, 4),
        "passes_required": passes_required,
        "passes_php_lint": passes_php_lint,
        "passes_phpcs": passes_phpcs,
        "passed": passes_required and passes_php_lint and passes_phpcs,
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

    for model_name in model_names:
        model_results = [result for result in results if result["model_name"] == model_name]
        fine = [result for result in model_results if result["variant"] == "fine_tuned"]
        base = [result for result in model_results if result["variant"] == "baseline"]

        fine_scores = [float(item["score"]) for item in fine]
        base_scores = [float(item["score"]) for item in base]
        fine_format_scores = [float(item.get("format_sanity", {}).get("score", 1.0)) for item in fine]
        base_format_scores = [float(item.get("format_sanity", {}).get("score", 1.0)) for item in base]
        all_fine_scores.extend(fine_scores)
        all_base_scores.extend(base_scores)

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
                }
            )

        fine_avg = _average(fine_scores)
        base_avg = _average(base_scores)
        per_model.append(
            {
                "model_name": model_name,
                "base_model": fine[0].get("base_model") if fine else (base[0].get("base_model") if base else ""),
                "fine_tuned_avg_score": fine_avg,
                "baseline_avg_score": base_avg,
                "delta_avg_score": round(fine_avg - base_avg, 4),
                "fine_tuned_pass_rate": _average([1.0 if item["passed"] else 0.0 for item in fine]),
                "baseline_pass_rate": _average([1.0 if item["passed"] else 0.0 for item in base]),
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
        "models": per_model,
    }


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
    tokenizer = auto_tokenizer_cls.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sample_outputs_dir = eval_dir / "sample_outputs"
    sample_outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest("evaluation", eval_dir)
    manifest.data["config"] = eval_cfg

    models_to_eval = _resolve_models_for_eval(config, eval_cfg)
    if not models_to_eval:
        logger.error("No models defined for evaluation.")
        return 1

    max_models = max(1, int(eval_cfg.get("max_models", 1)))
    models_to_eval = models_to_eval[:max_models]

    adapter_subdir = _adapter_subdir_for_mode(str(eval_cfg.get("mode", "test_run")))
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

        adapter_path = root / "models" / model_name / adapter_subdir / "adapter"
        if not adapter_path.exists():
            blocked_models.append(
                {
                    "model": model_name,
                    "reason": f"adapter not found at {adapter_path}",
                }
            )
            continue

        manifest.add_input(
            f"adapter_{model_name}",
            adapter_subdir,
            _hash_directory(adapter_path),
        )
        ready_models.append({"name": model_name, "base_model": base_model, "adapter_path": adapter_path})

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

        for variant in ["fine_tuned", "baseline"]:
            context = model.disable_adapter() if variant == "baseline" and hasattr(model, "disable_adapter") else nullcontext()
            with context:
                for prompt in prompt_suite:
                    prompt_id = str(prompt.get("id", "prompt"))
                    instruction = str(prompt.get("instruction", "")).strip()
                    input_text = str(prompt.get("input", ""))
                    category = str(prompt.get("category", "general"))

                    logger.info(f"Generating {variant} output for {model_name}:{prompt_id}")
                    output = _generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        input_text=input_text,
                        max_new_tokens=int(eval_cfg["max_new_tokens"]),
                        eval_cfg=eval_cfg,
                    )

                    checks, required = _required_checks_for_prompt(prompt_id, output)
                    external_checks = _run_external_checks(output, eval_cfg)
                    format_sanity = _compute_format_sanity(output)
                    score = _score_result(checks, required, external_checks)

                    output_path = model_dir / f"{variant}__{_sanitize_slug(prompt_id)}.txt"
                    with open(output_path, "w", encoding="utf-8") as handle:
                        handle.write(output)

                    result = {
                        "timestamp": _iso_timestamp(),
                        "model_name": model_name,
                        "base_model": base_model_id,
                        "variant": variant,
                        "prompt_id": prompt_id,
                        "category": category,
                        "instruction": instruction,
                        "input": input_text,
                        "output_file": output_path.relative_to(root).as_posix(),
                        "output_length": len(output),
                        "checks": checks,
                        "required_checks": required,
                        "external_checks": external_checks,
                        "format_sanity": format_sanity,
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
    metrics = {
        "stage": "evaluation",
        "timestamp": _iso_timestamp(),
        "seed": seed,
        "mode": eval_cfg.get("mode", "test_run"),
        "prompt_suite": prompt_suite,
        "blocked_models": blocked_models,
        "summary": summary,
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
            "blocked_models": len(blocked_models),
            "result_count": len(all_results),
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
    "_extract_code_blocks",
    "_required_checks_for_prompt",
    "_score_result",
]
