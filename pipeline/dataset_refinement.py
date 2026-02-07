import copy
import json
import random
import re
from pathlib import Path
from typing import Any

from .logger import PipelineLogger
from .manifest import Manifest, calculate_hash

RETRIEVAL_PROMPT_RE = re.compile(
    r"^Show me the implementation of the (?P<kind>class|interface|trait|enum) (?P<symbol>[A-Za-z_][A-Za-z0-9_]*) in the file (?P<path>.+)\.$"
)
VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
DECLARATION_RE = re.compile(r"\b(class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
NAMESPACE_RE = re.compile(r"^namespace\s+([^;]+);", re.MULTILINE)
USE_RE = re.compile(r"^use\s+([^;]+);", re.MULTILINE)
METHOD_RE = re.compile(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _load_split_records(split_path: Path, split_name: str) -> list[dict[str, Any]]:
    records = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["_source_split"] = split_name
            sample["_source_line"] = line_number
            records.append(sample)
    return records


def _is_test_sample(sample: dict[str, Any]) -> bool:
    source = str(sample.get("metadata", {}).get("source", "")).lower()
    return "/tests/" in source or source.endswith("test.php")


def _sample_source(sample: dict[str, Any]) -> str:
    source = str(sample.get("metadata", {}).get("source", "")).strip()
    if source:
        return source
    split_name = str(sample.get("_source_split", "unknown"))
    split_line = str(sample.get("_source_line", "0"))
    return f"__unknown_source__:{split_name}:{split_line}"


def _is_php_output(sample: dict[str, Any]) -> bool:
    output = str(sample.get("output", ""))
    return output.lstrip().startswith("<?php")


def _is_source_php_file(sample: dict[str, Any]) -> bool:
    source = _sample_source(sample).lower()
    return source.endswith((".php", ".module", ".inc", ".install", ".theme"))


def _is_unchunked_sample(sample: dict[str, Any]) -> bool:
    refinement = sample.get("metadata", {}).get("refinement", {})
    if not isinstance(refinement, dict):
        return True
    return int(refinement.get("chunk_total", 1)) <= 1


def _is_augmentation_candidate(sample: dict[str, Any]) -> tuple[bool, str]:
    if _is_test_sample(sample):
        return False, "augmentation_from_test_source"
    if not _is_source_php_file(sample):
        return False, "augmentation_non_php_source"
    if not _is_php_output(sample):
        return False, "augmentation_non_php_output"
    if not _is_unchunked_sample(sample):
        return False, "augmentation_chunked_source"
    return True, ""


def _detect_symbol_kind_and_name(sample: dict[str, Any]) -> tuple[str, str]:
    output = sample.get("output", "")
    declaration = DECLARATION_RE.search(output)
    if declaration:
        return declaration.group(1), declaration.group(2)

    source = str(sample.get("metadata", {}).get("source", ""))
    base = Path(source).name
    if base.endswith(".php"):
        symbol_name = base[:-4]
    else:
        symbol_name = base or "UnknownSymbol"

    lower_base = base.lower()
    if lower_base.endswith("interface.php"):
        return "interface", symbol_name
    if lower_base.endswith("trait.php"):
        return "trait", symbol_name
    if lower_base.endswith("enum.php"):
        return "enum", symbol_name
    return "class", symbol_name


def _validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    instruction = str(sample.get("instruction", "")).strip()
    source = str(sample.get("metadata", {}).get("source", "")).lower()
    prompt_match = RETRIEVAL_PROMPT_RE.match(instruction)

    if prompt_match:
        symbol_slot = prompt_match.group("symbol").strip()
        prompt_kind = prompt_match.group("kind").strip()
        if not VALID_SYMBOL_RE.match(symbol_slot):
            return False, "malformed_instruction_class_slot"
        detected_kind, _detected_name = _detect_symbol_kind_and_name(sample)
        if prompt_kind != detected_kind:
            return False, f"{prompt_kind}_{detected_kind}_mismatch"
    elif instruction.startswith("Show me the implementation of"):
        return False, "invalid_symbol_kind_prompt"

    instruction_lower = instruction.lower()
    if "implementation of the class" in instruction_lower:
        if source.endswith("interface.php"):
            return False, "class_interface_mismatch"
        if source.endswith("trait.php"):
            return False, "class_trait_mismatch"
        if source.endswith("enum.php"):
            return False, "class_enum_mismatch"
        kind, _name = _detect_symbol_kind_and_name(sample)
        if kind in {"interface", "trait", "enum"}:
            return False, f"class_{kind}_mismatch"

    return True, ""


def _chunk_sample(
    sample: dict[str, Any], max_output_lines: int, overlap_lines: int
) -> list[dict[str, Any]]:
    output = sample.get("output", "")
    lines = output.splitlines()
    if len(lines) <= max_output_lines:
        return [sample]

    overlap = max(0, min(overlap_lines, max_output_lines - 1))
    ranges = []
    start = 0
    while start < len(lines):
        end = min(start + max_output_lines, len(lines))
        ranges.append((start, end))
        if end == len(lines):
            break
        start = end - overlap

    chunked = []
    total = len(ranges)
    for idx, (start, end) in enumerate(ranges, start=1):
        record = copy.deepcopy(sample)
        chunk_output = "\n".join(lines[start:end])
        if output.endswith("\n") and end == len(lines):
            chunk_output += "\n"
        record["output"] = chunk_output
        base_instruction = str(sample.get("instruction", "")).strip()
        record["instruction"] = (
            f"{base_instruction} [Part {idx}/{total}, lines {start + 1}-{end}]"
        )
        metadata = dict(record.get("metadata", {}))
        refinement = dict(metadata.get("refinement", {}))
        refinement.update(
            {
                "chunk_index": idx,
                "chunk_total": total,
                "line_start": start + 1,
                "line_end": end,
            }
        )
        metadata["refinement"] = refinement
        record["metadata"] = metadata
        chunked.append(record)
    return chunked


def _build_spec(sample: dict[str, Any], symbol_kind: str, symbol_name: str) -> str:
    output = sample.get("output", "")
    source = str(sample.get("metadata", {}).get("source", ""))

    namespace_match = NAMESPACE_RE.search(output)
    namespace_value = namespace_match.group(1).strip() if namespace_match else "Drupal\\Custom"

    uses = USE_RE.findall(output)
    uses = uses[:8]
    methods = sorted(set(METHOD_RE.findall(output)))[:12]

    lines = [
        f"Source file: {source}",
        f"Target symbol: {symbol_kind} {symbol_name}",
        f"Namespace: {namespace_value}",
        "Requirements:",
        "- Use Drupal 11 and PHP 8.3 compatible patterns.",
        f"- Provide a complete {symbol_kind} implementation for {symbol_name}.",
    ]
    if uses:
        lines.append("- Include required imports:")
        for statement in uses:
            lines.append(f"  - {statement}")
    if methods:
        lines.append("- Ensure these methods exist where relevant:")
        for method in methods:
            lines.append(f"  - {method}()")
    return "\n".join(lines)


def _mutate_buggy_excerpt(output: str, excerpt_lines: int) -> str:
    lines = output.splitlines()
    excerpt = "\n".join(lines[: max(1, excerpt_lines)])
    if "declare(strict_types=1);" in excerpt:
        return excerpt.replace("declare(strict_types=1);", "declare(strict_types=0);", 1)
    if "public function " in excerpt:
        return excerpt.replace("public function ", "function ", 1)
    if "protected function " in excerpt:
        return excerpt.replace("protected function ", "function ", 1)
    if "#[" in excerpt:
        return excerpt.replace("#[", "[", 1)
    return excerpt + "\n// BUG: ensure this implementation follows Drupal 11 and PHP 8.3."


def _build_augmented_sample(
    sample: dict[str, Any], augmentation_type: str, excerpt_lines: int, max_output_lines: int
) -> dict[str, Any]:
    symbol_kind, symbol_name = _detect_symbol_kind_and_name(sample)
    source = str(sample.get("metadata", {}).get("source", ""))
    output = sample.get("output", "")

    metadata = copy.deepcopy(sample.get("metadata", {}))
    refinement = dict(metadata.get("refinement", {}))
    refinement["augmentation_type"] = augmentation_type
    metadata["refinement"] = refinement

    if augmentation_type == "bugfix":
        instruction = (
            f"Fix this broken Drupal 11 {symbol_kind} implementation and return corrected PHP code."
        )
        input_text = (
            f"Symbol: {symbol_name}\n"
            f"Source: {source}\n"
            "Broken snippet:\n"
            f"{_mutate_buggy_excerpt(output, excerpt_lines)}"
        )
        output_text = output
    elif augmentation_type == "refactor":
        instruction = (
            f"Refactor this Drupal 11 {symbol_kind} for readability and maintainability while preserving behavior."
        )
        input_text = (
            f"Symbol: {symbol_name}\n"
            f"Source: {source}\n"
            "Current implementation:\n"
            + "\n".join(output.splitlines()[: max(1, excerpt_lines)])
        )
        output_text = output
    elif augmentation_type == "write_from_spec":
        instruction = "Implement this Drupal 11 component from the specification. Return complete PHP code."
        input_text = _build_spec(sample, symbol_kind, symbol_name)
        output_text = output
    else:
        instruction = (
            "Explain the implementation plan in up to 3 bullet points, then provide the complete Drupal 11 PHP implementation."
        )
        input_text = _build_spec(sample, symbol_kind, symbol_name)
        output_text = (
            "- Follow Drupal 11 architecture and PHP 8.3 typing.\n"
            "- Preserve expected behavior and APIs from the specification.\n"
            "- Keep implementation concise and production-ready.\n\n"
            f"{output}"
        )

    output_lines = output_text.splitlines()
    if len(output_lines) > max_output_lines:
        output_text = "\n".join(output_lines[:max_output_lines])

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": metadata,
    }


def _rebalance_test_ratio(
    samples: list[dict[str, Any]], seed: int, target_test_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    tests = [sample for sample in samples if _is_test_sample(sample)]
    prod = [sample for sample in samples if not _is_test_sample(sample)]

    total_before = len(samples)
    test_ratio_before = (len(tests) / total_before) if total_before else 0.0

    if not tests or not prod:
        return samples, [], test_ratio_before, test_ratio_before
    if target_test_ratio <= 0:
        dropped = list(tests)
        return prod, dropped, test_ratio_before, 0.0
    if target_test_ratio >= 1:
        return samples, [], test_ratio_before, test_ratio_before

    max_tests = int((target_test_ratio * len(prod)) / (1.0 - target_test_ratio))
    max_tests = max(0, max_tests)
    if len(tests) <= max_tests:
        return samples, [], test_ratio_before, test_ratio_before

    rng = random.Random(seed)
    test_indices = list(range(len(tests)))
    rng.shuffle(test_indices)
    keep_idx = set(sorted(test_indices[:max_tests]))

    kept_tests = [tests[idx] for idx in range(len(tests)) if idx in keep_idx]
    dropped_tests = [tests[idx] for idx in range(len(tests)) if idx not in keep_idx]
    rebalanced = prod + kept_tests
    test_ratio_after = len(kept_tests) / len(rebalanced) if rebalanced else 0.0
    return rebalanced, dropped_tests, test_ratio_before, test_ratio_after


def _split_dataset(
    samples: list[dict[str, Any]], targets: dict[str, float], seed: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        source = _sample_source(sample)
        grouped.setdefault(source, []).append(sample)

    source_keys = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(source_keys)

    total = len(samples)
    train_target = float(targets.get("train", 0.8))
    valid_target = float(targets.get("valid", 0.1))
    target_counts = {
        "train": int(total * train_target),
        "valid": int(total * valid_target),
    }
    target_counts["test"] = total - target_counts["train"] - target_counts["valid"]

    counts = {"train": 0, "valid": 0, "test": 0}
    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}

    def score_assignment(split_name: str, group_size: int) -> tuple[int, int, int]:
        local_counts = dict(counts)
        local_counts[split_name] += group_size
        l1_error = sum(abs(local_counts[name] - target_counts[name]) for name in ("train", "valid", "test"))
        deficit = target_counts[split_name] - counts[split_name]
        return l1_error, -deficit, local_counts[split_name]

    for source in source_keys:
        group = grouped[source]
        group_size = len(group)
        choices = ("train", "valid", "test")
        best_split = min(choices, key=lambda name: score_assignment(name, group_size))
        assigned[best_split].extend(group)
        counts[best_split] += group_size

    for split_name in ("train", "valid", "test"):
        rng.shuffle(assigned[split_name])

    return assigned


def _derive_feedback_targets(eval_metrics_path: Path) -> dict[str, Any]:
    if not eval_metrics_path.exists():
        return {
            "source": None,
            "weak_categories": [],
            "details": [],
        }

    try:
        with open(eval_metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except Exception:
        return {
            "source": str(eval_metrics_path),
            "weak_categories": [],
            "details": [],
        }

    summary = metrics.get("summary", {})
    weak_details: dict[str, dict[str, Any]] = {}
    for model in summary.get("models", []):
        for delta in model.get("prompt_deltas", []):
            category = str(delta.get("category", "general"))
            delta_score = float(delta.get("delta", 0.0))
            if delta_score >= 0:
                continue
            item = weak_details.setdefault(category, {"count": 0, "avg_delta": 0.0})
            item["count"] += 1
            item["avg_delta"] += delta_score

    details = []
    for category, item in weak_details.items():
        avg_delta = item["avg_delta"] / max(1, item["count"])
        details.append(
            {
                "category": category,
                "negative_prompt_count": item["count"],
                "avg_delta": round(avg_delta, 4),
            }
        )
    details.sort(key=lambda item: item["avg_delta"])
    weak_categories = [item["category"] for item in details]
    return {
        "source": str(eval_metrics_path),
        "weak_categories": weak_categories,
        "details": details,
    }


def _score_candidate_for_feedback(sample: dict[str, Any], weak_categories: list[str]) -> int:
    if not weak_categories:
        return 0
    source = str(sample.get("metadata", {}).get("source", "")).lower()
    text = f"{source} {sample.get('instruction', '')}".lower()
    category_keywords = {
        "attributes": ["attribute", "plugin", "block", "#["],
        "di": ["service", "dependency", "container", "inject"],
        "routing": ["routing", "route", "controller", "path"],
        "sdc": ["component", "twig", "sdc"],
    }
    score = 0
    for category in weak_categories:
        keywords = category_keywords.get(category, [category])
        if any(keyword in text for keyword in keywords):
            score += 1
    return score


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            clean_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            handle.write(json.dumps(clean_sample, ensure_ascii=True) + "\n")


def _read_refinement_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    defaults = {
        "input_version": "v1",
        "output_version": "v2",
        "seed": seed,
        "max_output_lines": 300,
        "chunk_overlap_lines": 30,
        "target_test_ratio": 0.15,
        "exclude_test_sources_from_training_pool": True,
        "augmentation": {
            "enabled": True,
            "ratio": 0.75,
            "input_excerpt_lines": 120,
            "types": [
                "bugfix",
                "refactor",
                "write_from_spec",
                "explain_and_implement",
            ],
        },
    }
    merged = defaults | config.get("dataset_refinement", {})
    merged_augmentation = defaults["augmentation"] | merged.get("augmentation", {})
    merged["augmentation"] = merged_augmentation
    return merged


def run_dataset_refinement_stage(config: dict, logger: PipelineLogger, root: Path) -> int:
    refine_cfg = _read_refinement_config(config)
    dataset_root = root / "dataset"
    input_dir = dataset_root / refine_cfg["input_version"]
    output_dir = dataset_root / refine_cfg["output_version"]
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "train": input_dir / "train.jsonl",
        "valid": input_dir / "valid.jsonl",
        "test": input_dir / "test.jsonl",
    }
    for split_name, split_path in required_files.items():
        if not split_path.exists():
            logger.error(f"Input split missing for refinement: {split_path} ({split_name})")
            return 1

    manifest = Manifest("dataset_refinement", output_dir)
    for split_name, split_path in required_files.items():
        manifest.add_input(
            f"{refine_cfg['input_version']}_{split_name}",
            refine_cfg["input_version"],
            calculate_hash(split_path),
        )
    manifest.data["config"] = refine_cfg
    feedback_targets = _derive_feedback_targets(root / "eval" / "metrics.json")

    records = []
    for split_name, split_path in required_files.items():
        records.extend(_load_split_records(split_path, split_name))

    filtered_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}
    for sample in records:
        passed, reason = _validate_sample(sample)
        if passed:
            filtered_records.append(sample)
            continue
        rejected = copy.deepcopy(sample)
        rejected["rejection_reason"] = reason
        rejected_records.append(rejected)
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    max_output_lines = int(refine_cfg["max_output_lines"])
    overlap_lines = int(refine_cfg["chunk_overlap_lines"])
    chunked_records: list[dict[str, Any]] = []
    chunked_source_count = 0
    for sample in filtered_records:
        chunks = _chunk_sample(sample, max_output_lines=max_output_lines, overlap_lines=overlap_lines)
        if len(chunks) > 1:
            chunked_source_count += 1
        chunked_records.extend(chunks)

    exclude_test_sources = bool(refine_cfg.get("exclude_test_sources_from_training_pool", False))
    eval_candidate_pool: list[dict[str, Any]] = []
    if exclude_test_sources:
        eval_candidate_pool = [sample for sample in chunked_records if _is_test_sample(sample)]
        rebalanced_records = [sample for sample in chunked_records if not _is_test_sample(sample)]
        dropped_tests: list[dict[str, Any]] = []
        test_ratio_before = len(eval_candidate_pool) / len(chunked_records) if chunked_records else 0.0
        test_ratio_after = 0.0
    else:
        rebalanced_records, dropped_tests, test_ratio_before, test_ratio_after = _rebalance_test_ratio(
            chunked_records,
            seed=int(refine_cfg["seed"]) + 17,
            target_test_ratio=float(refine_cfg["target_test_ratio"]),
        )
        for dropped in dropped_tests:
            rejected = copy.deepcopy(dropped)
            rejected["rejection_reason"] = "test_rebalance_drop"
            rejected_records.append(rejected)
        if dropped_tests:
            rejection_reasons["test_rebalance_drop"] = rejection_reasons.get("test_rebalance_drop", 0) + len(
                dropped_tests
            )

    augmentation_cfg = refine_cfg["augmentation"]
    augmented_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    if augmentation_cfg.get("enabled", True):
        for sample in rebalanced_records:
            can_augment, reason = _is_augmentation_candidate(sample)
            if can_augment:
                candidate_records.append(sample)
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        ratio = max(0.0, min(1.0, float(augmentation_cfg.get("ratio", 0.75))))
        candidate_count = int(len(candidate_records) * ratio)
        rng = random.Random(int(refine_cfg["seed"]) + 53)
        candidate_indices = list(range(len(candidate_records)))
        rng.shuffle(candidate_indices)
        weak_categories = feedback_targets.get("weak_categories", [])
        candidate_indices.sort(
            key=lambda idx: (
                _score_candidate_for_feedback(candidate_records[idx], weak_categories),
                -idx,
            ),
            reverse=True,
        )
        selected_indices = sorted(candidate_indices[:candidate_count])
        aug_types = augmentation_cfg.get(
            "types",
            ["bugfix", "refactor", "write_from_spec", "explain_and_implement"],
        )
        if not aug_types:
            aug_types = ["write_from_spec"]
        excerpt_lines = int(augmentation_cfg.get("input_excerpt_lines", 120))

        for idx, selected_index in enumerate(selected_indices):
            source_sample = candidate_records[selected_index]
            augmentation_type = aug_types[idx % len(aug_types)]
            augmented_records.append(
                _build_augmented_sample(
                    source_sample,
                    augmentation_type=augmentation_type,
                    excerpt_lines=excerpt_lines,
                    max_output_lines=max_output_lines,
                )
            )

    final_records = rebalanced_records + augmented_records
    split_targets = config.get("dataset", {}).get("targets", {"train": 0.8, "valid": 0.1, "test": 0.1})
    splits = _split_dataset(final_records, split_targets, seed=int(refine_cfg["seed"]) + 101)

    for split_name, split_samples in splits.items():
        split_path = output_dir / f"{split_name}.jsonl"
        _write_jsonl(split_path, split_samples)
        manifest.add_output(
            f"{split_name}_split",
            f"dataset/{refine_cfg['output_version']}/{split_name}.jsonl",
            calculate_hash(split_path),
        )

    rejected_path = output_dir / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected_records)
    manifest.add_output(
        "rejected_samples",
        f"dataset/{refine_cfg['output_version']}/rejected.jsonl",
        calculate_hash(rejected_path),
    )

    eval_candidate_path = output_dir / "eval_candidate_pool.jsonl"
    _write_jsonl(eval_candidate_path, eval_candidate_pool)
    manifest.add_output(
        "eval_candidate_pool",
        f"dataset/{refine_cfg['output_version']}/eval_candidate_pool.jsonl",
        calculate_hash(eval_candidate_path),
    )

    sample_type_distribution: dict[str, int] = {"retrieval": 0}
    for sample in augmented_records:
        aug_type = sample.get("metadata", {}).get("refinement", {}).get("augmentation_type", "unknown")
        sample_type_distribution[aug_type] = sample_type_distribution.get(aug_type, 0) + 1
    sample_type_distribution["retrieval"] = len(rebalanced_records)

    yield_breakdown = {
        "input_records": len(records),
        "after_validation_filter": len(filtered_records),
        "after_chunking": len(chunked_records),
        "after_test_pool_extraction": len(rebalanced_records),
        "after_augmentation": len(final_records),
        "rejected_records": len(rejected_records),
    }

    metrics = {
        "input_total": len(records),
        "filtered_total": len(filtered_records),
        "chunked_source_records": chunked_source_count,
        "chunked_total_records": len(chunked_records),
        "rebalanced_total_records": len(rebalanced_records),
        "dropped_test_records": len(dropped_tests),
        "augmented_records": len(augmented_records),
        "augmentation_candidate_pool": len(candidate_records) if augmentation_cfg.get("enabled", True) else 0,
        "final_total": len(final_records),
        "test_ratio_before_rebalance": round(test_ratio_before, 4),
        "test_ratio_after_rebalance": round(test_ratio_after, 4),
        "split_train": len(splits["train"]),
        "split_valid": len(splits["valid"]),
        "split_test": len(splits["test"]),
        "eval_candidate_pool_size": len(eval_candidate_pool),
        "exclude_test_sources_from_training_pool": exclude_test_sources,
        "yield_breakdown": yield_breakdown,
        "feedback_targets": feedback_targets,
        "rejection_reasons": rejection_reasons,
        "sample_type_distribution": sample_type_distribution,
    }
    manifest.set_metrics(metrics)
    manifest.save()

    index_path = dataset_root / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    index[refine_cfg["output_version"]] = {
        "timestamp": manifest.data["timestamp"],
        "stage": "dataset_refinement",
        "metrics": {
            "train": len(splits["train"]),
            "valid": len(splits["valid"]),
            "test": len(splits["test"]),
            "total": len(final_records),
        },
        "path": f"dataset/{refine_cfg['output_version']}",
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    logger.info(
        "Dataset refinement complete. "
        f"Input={len(records)} Final={len(final_records)} "
        f"Train/Valid/Test={len(splits['train'])}/{len(splits['valid'])}/{len(splits['test'])}"
    )
    return 0
