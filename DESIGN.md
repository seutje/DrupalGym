# Design Document: Automated Drupal 11 Training Pipeline for Qwen2.5-Coder and Ministral-3

## Overview
This document specifies a reproducible, automated training pipeline that turns Drupal 11 source/Docs into supervised fine-tuning (SFT) and evaluation datasets, trains QLoRA adapters compatible with both **Qwen2.5-Coder (7B)** and **Ministral-3 (8B)**, and supports **consumer-grade hardware** plus a single rented **L40S** for full-scale training on current dataset volumes. Future larger-scale runs are tied to improvements in data-gathering throughput and quality. The design prioritizes data quality, automation, traceability, and model-agnostic dataset formats described in the research summary. 【F:RESEARCH.md†L1-L496】

## Goals
- **Automate data gathering and curation** for Drupal 11–specific code and documentation.
- **Provide a single pipeline** that can train and evaluate adapters for **Qwen2.5-Coder** and **Ministral-3**.
- **Support consumer hardware** (e.g., RTX 4070) for short test runs and small datasets, then run full-scale training on **single-GPU L40S** for current dataset volumes.
- **Ensure reproducibility** via versioned manifests, dataset hashes, and pipeline configuration.

## Non-Goals
- Shipping a full MLOps platform (Kubernetes, multi-tenant serving).
- Fine-tuning larger than 8B models.
- Replacing upstream documentation or training with proprietary data not described in the research.

## Key Constraints & Assumptions
- Targeting **Drupal 11 + PHP 8.3**, emphasizing attribute-based plugin discovery and Symfony 7 APIs as critical differentiators. 【F:RESEARCH.md†L63-L156】
- **Data quality is the primary bottleneck**; automation must include deduplication and quality filtering. 【F:RESEARCH.md†L156-L238】
- **QLoRA/PEFT** is required for consumer hardware feasibility. 【F:RESEARCH.md†L238-L327】
- Dataset format should be **model-agnostic** (JSONL with instruction/input/output).

---

## Pipeline Architecture (High-Level)

```
[Source Discovery]
  -> [Acquisition]
    -> [Normalization + Dedup]
      -> [Instruction + Synthetic Generation]
        -> [Quality Gates]
          -> [Dataset Packaging]
            -> [Training (QLoRA)]
              -> [Evaluation]
                -> [Export + Quantization]
```

### 1) Source Discovery (Automated)
**Purpose:** Identify Drupal 11–compatible repositories and documentation sources.

**Inputs**
- Drupal core repository (stable Drupal 11 branch).
- Contrib module list filtered by `composer.json` constraints (`^11`).
- Drupal API/Docs + Symfony 7 docs.
- Security advisories and API change notices.

**Automation**
- **Crawler + filter**:
  - Scrape module metadata from Drupal.org project API.
  - Filter by `composer.json` constraint `"drupal/core": "^11"`.
- **Doc crawler**:
  - Pull docs and API pages (rate-limited, stored with source URL).

**Artifacts**
- `sources/manifest.json` with repo URLs, commit refs, doc URLs, and timestamps.

---

### 2) Acquisition (Automated)
**Purpose:** Fetch sources and store raw snapshots.

**Automation**
- `git clone` or `git fetch` for code repos.
- HTTP fetch for docs (API pages, security advisories, changelogs).
- Store raw HTML/markdown and raw code in a staging directory.

**Artifacts**
- `raw/code/<repo>/...`
- `raw/docs/<domain>/<path>.html`
- `raw/manifest.json`

---

### 3) Normalization + Dedup (Automated)
**Purpose:** Clean noise and remove boilerplate duplication.

**Normalization Steps**
- Strip license headers and common banners.
- Normalize whitespace and line endings.
- Canonicalize PHP attribute syntax and remove trailing whitespace.

**Dedup Strategy**
- Hash files post-normalization.
- Deduplicate identical or near-identical templates (e.g., boilerplate info.yml).

**Artifacts**
- `clean/code/...`
- `clean/docs/...`
- `clean/dedup_manifest.json`

---

### 4) Instruction + Synthetic Generation (Automated)
**Purpose:** Convert raw code/docs into instruction pairs for SFT.

**Instruction Templates**
- “Create a Drupal 11 plugin using PHP 8.3 attributes.”
- “Refactor procedural \Drupal::service() usage to constructor injection.”
- “Explain differences between Drupal 10/11 plugin discovery.”

**Synthetic Data**
- Teacher model (e.g., Gemini 3 Flash) generates instruction/response pairs from docs.
- Model outputs are validated via static analysis (PHPStan/PHPCS) and unit tests where possible. 【F:RESEARCH.md†L238-L396】

**Artifacts**
- `sft/instructions.jsonl`
- `sft/synthetic.jsonl`
- `sft/combined.jsonl`

---

### 5) Quality Gates (Automated)
**Purpose:** Ensure training samples are high-signal and Drupal 11 compliant.

**Checks**
- PHPCS (Drupal standard) for code style.
- PHPStan-Drupal to catch deprecated/invalid APIs.
- Rule-based filters for PHP 8.3 attribute usage and dependency injection.

**Artifacts**
- `quality/report.json`
- `quality/rejected.jsonl`

---

### 6) Dataset Packaging
**Purpose:** Produce a versioned dataset for both models.

**Output**
- `dataset/v1/train.jsonl`
- `dataset/v1/valid.jsonl`
- `dataset/v1/test.jsonl`
- `dataset/v1/manifest.json` (hashes, sources, stats)

**Compatibility**
- **Format:** Alpaca-style JSONL with `instruction`, `input`, `output` fields.
- **Tokenization:** defer to model-specific tokenizers at training time.

---

### 7) Training (QLoRA)
**Purpose:** Train LoRA adapters for Qwen2.5-Coder and Ministral-3.

**Consumer Hardware Test Runs**
- Small dataset (1k–5k samples), sequence length 2k.
- QLoRA with 4-bit base model.
- Framework: **Unsloth** for speed/VRAM efficiency. 【F:RESEARCH.md†L281-L315】

**Full Training on L40S (Current Baseline)**
- Current dataset volumes with moderate context length (~2k–4k).
- QLoRA config tuned for single-GPU stability and reproducibility.
- Framework: Unsloth or Axolotl-compatible setup.

**Model Compatibility Notes**
- Use a shared dataset and per-model config:
  - Qwen2.5-Coder: GQA, 128k context; tuned for code completion.
  - Ministral-3: sliding window/GQA, 262k context; tuned for reasoning.
- LoRA adapters stored in per-model directories.

---

### 8) Evaluation
**Purpose:** Validate generated code quality and Drupal 11 standards.

**Automated Metrics**
- PHPCS + PHPStan-Drupal on generated code.
- Targeted prompt suite for Drupal 11 tasks (attributes, DI, SDC, routing).
- Security checks (input sanitization, parameterized queries).

**Artifacts**
- `eval/metrics.json`
- `eval/sample_outputs/`

---

### 9) Export + Quantization
**Purpose:** Export LoRA adapters and optionally merge for deployment.

**Output**
- LoRA adapters (`.safetensors`).
- Optional merged model for GGUF/EXL2 for local inference. 【F:RESEARCH.md†L346-L371】

---

## Automation Details

### Data Gathering Automation
1. **Discover** Drupal 11 compatible modules:
   - Use Drupal.org project API or scrape module metadata.
   - Filter `composer.json` for `drupal/core: ^11`.
2. **Acquire** repositories:
   - Shallow clone with pinned commits.
3. **Collect docs**:
   - Crawl Drupal API change notices, security advisories, Symfony 7 docs.

### Pipeline Orchestrator
- A single `pipeline.yaml` defines sources, filters, dataset size targets, and model configs.
- Each stage outputs a manifest with hashes to guarantee reproducibility.

### Reproducibility
- Every dataset version includes:
  - Source list with commit hashes.
  - Normalization + dedup rules.
  - Random seed used for train/val/test split.

---

## Hardware Strategy

### Consumer-Grade (Iterative Testing)
- RTX 3090/4090 recommended (24GB VRAM), should support initial testing on 4070 (12GB).
- QLoRA 4-bit, seq len 2048.
- 500–10k samples; 1–3 epochs. 【F:RESEARCH.md†L324-L383】

### L40S (Full Training Baseline)
- Stable single-GPU full-scale runs for current dataset sizes.
- Moderate context lengths (2k–4k) with QLoRA.
- Use this as the default production training target.

### Future Larger-Scale Training
- Move to larger GPUs and longer contexts only after improving data gathering automation to produce larger, cleaner datasets.
- Target expansion through source discovery limits, dedup quality, and augmentation correctness.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Low-quality or outdated data | Model generates deprecated Drupal 10 patterns | Strict composer.json filtering; PHP 8.3 attribute checks; curated docs. |
| Dataset duplication | Overfitting on boilerplate patterns | Hash-based dedup; template filtering. |
| Consumer GPU OOM | Failed test runs | QLoRA, smaller seq length, gradient checkpointing. |
| Model divergence between Qwen/Ministral | Inconsistent quality | Shared dataset + per-model LoRA configs. |

---

## Implementation Plan (Phased)

1. **Phase 1: Data automation**
   - Build crawler + repo fetcher.
   - Implement normalization + dedup.
2. **Phase 2: SFT generation**
   - Template-based instructions + teacher model generation.
   - Static analysis filters (PHPCS/PHPStan).
3. **Phase 3: Training pipeline**
   - Unsloth-based QLoRA configs for both models.
   - Consumer test runs.
4. **Phase 4: Scale-up after data pipeline improvements**
   - Improve source coverage and dataset quality first.
   - Then increase dataset size/context and run full evaluation suite.

---

## Acceptance Criteria
- Fully automated data gathering with Drupal 11 filtering.
- SFT dataset format compatible with **Qwen2.5-Coder** and **Ministral-3**.
- Successful test runs on consumer GPU with QLoRA.
- Repeatable full training on single-GPU L40S with documented configs and reproducible datasets.
