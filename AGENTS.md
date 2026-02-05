# Agent Instructions for DrupalGym

## Project Goal
Build an automated, reproducible training pipeline that turns Drupal 11 code/docs into SFT + eval datasets, then trains QLoRA adapters for Qwen2.5-Coder (7B) and Ministral-3 (8B) with consumer-GPU test runs and H100 scaling.

## Core Principles
- Prioritize data quality, traceability, and reproducibility.
- Keep dataset format model-agnostic (JSONL with `instruction`, `input`, `output`).
- Drupal 11 + PHP 8.3 compliance is non-negotiable.
- Automate everything possible; keep manual steps limited to explicit verification gates.

## Work Phasing (Follow PLAN.md)
Execute work in phases 0–11 as defined in `PLAN.md`. Each phase is:
- Fully executed by the agent.
- Followed by a **single human verification step** before moving to the next phase.

If a phase cannot be completed due to missing dependencies, report:
- The exact blocker.
- What was attempted.
- The minimal next action to unblock.

## Required Artifacts by Stage
Maintain the directory structure and outputs referenced in `DESIGN.md` and `PLAN.md`:
- `sources/manifest.json`
- `raw/manifest.json`
- `clean/dedup_manifest.json`
- `sft/*.jsonl` (instructions, synthetic, combined)
- `quality/report.json` and `quality/rejected.jsonl`
- `dataset/vX/{train,valid,test}.jsonl` and `dataset/vX/manifest.json`
- `eval/metrics.json` and `eval/sample_outputs/`
- `models/<model>/{test_run,final}/`
- `manifests/` or equivalent for run metadata

Do not rename these without updating all references and documenting the change. Update the .gitignore file when necessary.

## Data and Quality Gates
Always enforce:
- Drupal 11 filtering (e.g., `composer.json` constraint `drupal/core: ^11`).
- PHP 8.3 attribute-based patterns and modern DI usage.
- Deduplication of boilerplate and near-duplicate templates.
- Code quality checks (PHPCS Drupal standard, PHPStan-Drupal where applicable).

Rejected samples must include a reason in `quality/rejected.jsonl`.

## Reproducibility Requirements
Every stage must emit:
- A manifest with input sources, hashes, and timestamps.
- A deterministic random seed (recorded in manifest or `pipeline.yaml`).
- Idempotent behavior when rerun with unchanged inputs.

Prefer a single `pipeline.yaml` as the source of truth for:
- Sources, filters, dataset sizes, model configs, and seeds.

## Training Guidance
Use QLoRA/PEFT. For consumer GPU tests:
- Small dataset (1k–5k samples), seq len ~2k, 4-bit base.
For H100:
- Full dataset (50k–100k), longer contexts (8k–32k), larger batch sizes.

Store adapters per model and per run.

## Communication Style for This Repo
When you finish a phase, summarize:
- What changed.
- Where outputs were written (file paths).
- What needs human verification.

Avoid speculative steps outside the current phase unless asked.
