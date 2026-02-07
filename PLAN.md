# Implementation Plan: Automated Drupal 11 Training Pipeline

This plan is structured for an agentic workflow: the agent executes all steps in each phase, then pauses for a single human verification step before proceeding.

## Phase 0: Project Scaffolding and Config
Goal: establish the pipeline skeleton, configuration, and reproducibility scaffolding.

Steps
1. Define the workspace layout for `raw/`, `clean/`, `sft/`, `quality/`, `dataset/`, `eval/`, `models/`, and `manifests/`.
2. Create a single `pipeline.yaml` with source definitions, filters, dataset targets, model configs, and random seeds.
3. Add versioned manifest format (JSON schema or documented keys) for each stage output.
4. Implement a common logging and metrics format (JSONL) for stage runs.
5. Add a minimal CLI entrypoint (e.g., `python -m pipeline run <stage>` or equivalent).

Human verification
- Verify the directory structure, `pipeline.yaml` contents, and manifest format match the design and are understandable/reviewable.

## Phase 1: Source Discovery
Goal: automatically discover Drupal 11–compatible code and documentation sources.

Steps
1. Implement Drupal.org project API client and metadata scraper.
2. Filter projects by `composer.json` constraint `drupal/core: ^11`.
3. Add curated sources for Drupal core, official docs/API, Symfony 7 docs, and security advisories.
4. Output `sources/manifest.json` with URLs, commit refs (if known), and timestamps.
5. Add rate limiting and retry policy for web sources.

Human verification
- Inspect a sampled `sources/manifest.json` for completeness, correctness, and Drupal 11 filtering accuracy.

## Phase 2: Acquisition
Goal: fetch and snapshot all sources reliably.

Steps
1. Implement git fetch/clone for repositories with pinned commits.
2. Implement HTTP fetch for docs and advisories, storing raw HTML/Markdown.
3. Normalize file naming for doc storage by domain/path.
4. Produce `raw/manifest.json` with checksums and fetch metadata.
5. Add idempotency: re-running acquisition does not re-download unchanged sources.

Human verification
- Validate that a sample repo and doc set were captured correctly and are reproducible with the manifest.

## Phase 3: Normalization and Deduplication
Goal: standardize inputs and remove duplicates/boilerplate.

Steps
1. Implement normalization rules for PHP files and docs (whitespace, line endings, license header stripping).
2. Add canonicalization for PHP 8.3 attributes where needed.
3. Hash normalized content and deduplicate exact matches.
4. Add near-duplicate detection for boilerplate templates (e.g., info.yml).
5. Produce `clean/dedup_manifest.json` with before/after counts and mappings.

Human verification
- Review a small set of normalized files and the dedup manifest to confirm expected reductions without data loss.

## Phase 4: Instruction and Synthetic Generation
Goal: convert cleaned sources into SFT instruction pairs and synthetic samples.

Steps
1. Implement instruction template generation from code and docs.
2. Add a teacher-model generation step for synthetic samples with strict prompt templates.
3. Validate model outputs with lightweight static checks (syntax, minimal lint, or unit stubs where feasible).
4. Merge template-based and synthetic outputs into `sft/combined.jsonl`.
5. Track provenance per sample (source file + transformation + generator).

Human verification
- Sample 20–50 entries across templates and synthetic outputs for correctness, Drupal 11 specificity, and clarity.

## Phase 5: Quality Gates
Goal: enforce Drupal 11 compliance and code quality before packaging.

Steps
1. Run PHPCS (Drupal standard) on code samples where applicable.
2. Run PHPStan-Drupal for API validation and deprecation checks.
3. Enforce rules for attribute-based plugin discovery and DI patterns.
4. Filter and log rejected samples into `quality/rejected.jsonl` with reasons.
5. Output `quality/report.json` with pass rates and error distributions.

Human verification
- Review `quality/report.json` and a random subset of rejections for proper filtering and no over-pruning.

## Phase 6: Dataset Packaging
Goal: produce versioned, model-agnostic datasets with manifests.

Steps
1. Split `sft/combined.jsonl` into train/valid/test with fixed seeds.
2. Emit `dataset/vX/train.jsonl`, `valid.jsonl`, `test.jsonl`.
3. Create `dataset/vX/manifest.json` with hashes, source lists, and stats.
4. Add dataset validation checks (schema, empty fields, length distribution).
5. Record dataset version in a top-level index file.

Human verification
- Confirm dataset counts, splits, and manifest hash integrity for the target version.

## Phase 7: Training (Consumer Hardware)
Goal: validate the training path on a small dataset using QLoRA.

Steps
1. Implement QLoRA config for Qwen2.5-Coder and Ministral-3 (small dataset, short context).
2. Add training scripts with consistent logging and checkpointing.
3. Run a short test (1k–5k samples) for each model.
4. Capture training metrics and loss curves.
5. Store LoRA adapters under `models/<model>/test_run/`.

Human verification
- Review training logs and confirm loss decreases and checkpoints are usable for inference.

## Phase 8: Evaluation
Goal: validate model outputs against Drupal 11 tasks and code quality.

Steps
1. Implement a targeted prompt suite (attributes, DI, routing, SDC).
2. Run automated code checks on generated outputs where applicable.
3. Compute and store metrics in `eval/metrics.json`.
4. Save representative outputs in `eval/sample_outputs/`.
5. Compare baseline vs. fine-tuned adapter behavior.

Human verification
- Review evaluation samples and metrics for meaningful improvement and Drupal 11 compliance.

## Phase 9: Full-Scale Training (L40S Baseline)
Goal: run stable full-scale single-GPU training on current dataset volume.

Steps
1. Update QLoRA configs for L40S-safe context and batch settings.
2. Execute training on a single L40S with the full available dataset.
3. Validate checkpoints and adapter integrity.
4. Run the full evaluation suite.
5. Store final adapters under `models/<model>/final/` with version tags.

Human verification
- Confirm final training artifacts, evaluation metrics, and reproducibility via manifests.
- Confirm readiness for future larger-scale runs only after data-gathering pipeline improvements increase dataset volume and quality.

## Phase 10: Export and Quantization
Goal: export adapters and optional merged models for deployment.

Steps
1. Export LoRA adapters as `.safetensors`.
2. Optionally merge adapters into base models for deployment.
3. Quantize to GGUF/EXL2 (if required for local inference).
4. Validate inference on a small prompt suite.
5. Update release notes and dataset/model version index.

Human verification
- Validate exported artifacts load successfully and produce expected outputs on sample prompts.

## Phase 11: Automation and Reproducibility Hardening
Goal: make the pipeline repeatable and robust.

Steps
1. Add end-to-end pipeline runner with stage caching.
2. Add CI tasks for linting, minimal validation, and schema checks.
3. Implement deterministic seed handling and environment capture.
4. Add documentation for rerunning and auditing pipeline stages.
5. Add alerting or failure summaries for long runs.

Human verification
- Run a full dry-run pipeline and verify the manifests, outputs, and logs are coherent and reproducible.
