# DrupalGym

Automated training pipeline for Drupal 11 AI models.

## Setup

1. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Prerequisites
Python 3.10+ is recommended. You also need `git` available on PATH for acquisition.

For training (stage 7), a CUDA-capable GPU is required. The current training run is a short test run tuned for consumer GPUs; it will fail fast if CUDA is not available.

## Configuration
Pipeline behavior is controlled by `pipeline.yaml`. Key knobs include:
`sources` (Drupal.org API discovery + curated sources), `dataset.targets` (train/valid/test split), `seed`, and the `models` list.

## Usage

The pipeline is managed via a CLI entrypoint.

### Run a Pipeline Stage
```bash
python3 -m pipeline run <stage_number>
```

Example:
```bash
python3 -m pipeline run 0
```

### Run the Default Pipeline Sequence
```bash
python3 -m pipeline pipeline
```

The default sequence is: `sources`, `acquisition`, `normalization`, `sft`, `quality`, `dataset`, `train`, `eval`.

## Stage Map
Stage names and aliases accepted by the CLI:

| Stage | Aliases |
| --- | --- |
| 0 | `phase0` |
| 1 | `phase1`, `sources` |
| 2 | `phase2`, `acquisition` |
| 3 | `phase3`, `normalization` |
| 4 | `phase4`, `sft` |
| 5 | `phase5`, `quality` |
| 6 | `phase6`, `dataset` |
| 7 | `phase7`, `train` |
| 8 | `phase8`, `eval` |
| 9 | `phase9`, `full_train` |
| 10 | `phase10`, `export` |
| 11 | `phase11`, `hardening` |

## Stage-by-Stage Instructions

**Stage 0: Project Scaffolding (no-op in CLI)**
Command:
```bash
python3 -m pipeline run 0
```
What it does: logs that scaffolding is already handled by existing files and directories.
Outputs: none.

**Stage 1: Source Discovery**
Command:
```bash
python3 -m pipeline run 1
```
Prereqs: network access to Drupal.org and other curated sources.
What it does: queries the Drupal.org project API, filters for Drupal 11 compatibility, and writes a consolidated manifest.
Outputs: `sources/manifest.json`.

**Stage 2: Acquisition**
Command:
```bash
python3 -m pipeline run 2
```
Prereqs: `sources/manifest.json`, network access, `git` on PATH.
What it does: clones or fetches repos into `raw/repos/` and crawls selected docs into `raw/docs/`.
Outputs: `raw/manifest.json`, `raw/repos/`, `raw/docs/`.

**Stage 3: Normalization and Deduplication**
Command:
```bash
python3 -m pipeline run 3
```
Prereqs: `raw/manifest.json`.
What it does: normalizes text and code, strips boilerplate, converts HTML docs to Markdown, and deduplicates.
Outputs: cleaned files under `clean/` and `clean/dedup_manifest.json`.

**Stage 4: SFT Generation**
Command:
```bash
python3 -m pipeline run 4
```
Prereqs: `clean/` data.
What it does: generates instruction samples from code and docs using template logic.
Outputs: `sft/combined.jsonl`.

**Stage 5: Quality Gates**
Command:
```bash
python3 -m pipeline run 5
```
Prereqs: `sft/combined.jsonl`.
What it does: applies heuristics to filter low-quality or irrelevant samples.
Outputs: `quality/passed.jsonl`, `quality/rejected.jsonl`, `quality/report.json`.

**Stage 6: Dataset Packaging**
Command:
```bash
python3 -m pipeline run 6
```
Prereqs: `quality/passed.jsonl`.
What it does: shuffles with a fixed seed and splits into train/valid/test.
Outputs: `dataset/v1/train.jsonl`, `dataset/v1/valid.jsonl`, `dataset/v1/test.jsonl`, `dataset/index.json`.

**Stage 7: Training (Consumer GPU Test Run)**
Command:
```bash
python3 -m pipeline run 7
```
Prereqs: CUDA-capable GPU, `dataset/v1/`.
What it does: runs a short QLoRA test training run for each model in `pipeline.yaml` with `max_steps=10`.
Outputs: adapters under `models/<model>/test_run/adapter/` and logs in `manifests/`.

**Stage 8: Evaluation**
Command:
```bash
python3 -m pipeline run 8
```
Prereqs: none (currently simulated outputs).
What it does: runs a small prompt suite and writes placeholder outputs and metrics.
Outputs: `eval/metrics.json`, `eval/sample_outputs/*.txt`.

**Stage 9: Full-Scale Training (placeholder)**
Command:
```bash
python3 -m pipeline run 9
```
What it does: currently logs a placeholder message only.
Outputs: none yet.

**Stage 10: Export and Quantization (placeholder)**
Command:
```bash
python3 -m pipeline run 10
```
What it does: currently logs a placeholder message only.
Outputs: none yet.

**Stage 11: Automation and Hardening (placeholder)**
Command:
```bash
python3 -m pipeline run 11
```
What it does: currently logs a placeholder message only.
Outputs: none yet.

## Project Structure
- `pipeline/`: Core logic and CLI implementation.
- `pipeline.yaml`: Pipeline configuration.
- `raw/`, `clean/`, `sft/`, `quality/`, `dataset/`: Data processing stages.
- `manifests/`: Logs and execution manifests.
- `models/`: Trained model adapters.
