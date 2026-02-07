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

The default sequence is: `sources`, `acquisition`, `normalization`, `sft`, `quality`, `dataset`, `dataset_refine`, `train`, `eval`.

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
| 6b | `phase6b`, `dataset_refine`, `refine` |
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

**Stage 6b: Dataset Refinement for Training**
Command:
```bash
python3 -m pipeline run 6b
```
Prereqs: `dataset/v1/`.
What it does: filters malformed retrieval prompts and class/interface/trait mismatches, chunks long outputs, rebalances test-vs-production samples, and adds non-retrieval instruction variants (`bugfix`, `refactor`, `write_from_spec`, `explain_and_implement`).
Outputs: `dataset/v2/train.jsonl`, `dataset/v2/valid.jsonl`, `dataset/v2/test.jsonl`, `dataset/v2/rejected.jsonl`, `dataset/v2/manifest.json`.

**Stage 7: Training (Consumer GPU Test Run)**
Command:
```bash
python3 -m pipeline run 7
```
Prereqs: CUDA-capable GPU, dataset configured by `dataset.training_version` (default `dataset/v2/`).
What it does: runs a QLoRA test from `training.test_run` in `pipeline.yaml` (defaults: `Qwen2.5-Coder-3B`, `max_seq_len=2048`, `max_steps=100`, ~10GB VRAM).
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

**Stage 10: Export and Quantization**
Command:
```bash
python3 -m pipeline run 10
```
Prereqs: trained adapters under `models/<model>/{test_run|final}/adapter/`, and llama.cpp conversion tools available on PATH or configured in `pipeline.yaml` under `export.tools`.
What it does: exports adapters as safetensors, optionally merges into base models, converts merged weights to `model-f16.gguf`, and generates configured GGUF quantizations (for example `Q4_K_M`, `Q8_0`).
Outputs: files under `models/<model>/{test_run|final}/exported/` including `model-f16.gguf` and quantized GGUF variants.

**Stage 11: Automation and Hardening (placeholder)**
Command:
```bash
python3 -m pipeline run 11
```
What it does: currently logs a placeholder message only.
Outputs: none yet.

## RunPod Deployment

To run a full training session (Phase 9) on a RunPod H100 instance while minimizing costs and time, follow these instructions.

### 1. Strategy: Data Preparation vs. Training
To minimize expensive H100 time, you should prepare the dataset on a cheaper instance or locally.

*Note: Per user requirements, we assume the entire pipeline (Stages 0-8) runs on the rented RunPod environment, while Stage 9 is the target for the full-scale H100 session.*

*   **Option A: Shared Volume (Recommended)**
    1.  Rent a cheap "Standard" instance (e.g., 1x RTX 3060 or a CPU-only instance) with a large network volume (e.g., 100GB).
    2.  Run stages 0 through 6 to produce the final dataset.
    3.  Stop the instance but **keep the volume**.
    4.  Deploy a "Secure Cloud" H100 instance and **attach the same volume**.
    5.  Run stage 9 (Full-Scale Training).

*   **Option B: Local Preparation & Upload**
    1.  Run stages 0 through 6 on your local machine.
    2.  Copy the following data to the RunPod instance via SCP or the RunPod web terminal:
        *   `dataset/` (The entire versioned dataset directory)
        *   `pipeline.yaml`
        *   `pipeline/`
        *   `requirements.txt`
        *   `scripts/runpod_setup.sh`

### 2. Setup on RunPod
Once logged into your H100 instance:

```bash
# Clone the repository (if not already copied)
git clone <your-repo-url> drupalgym
cd drupalgym

# Run the setup script
bash scripts/runpod_setup.sh

# Enter the virtual environment
source venv/bin/activate
```

### 3. Running the 20-Hour Training
The `pipeline.yaml` is configured with `full_scale` targets for `Qwen2.5-Coder-7B`. To execute the 20-hour session:

```bash
python3 -m pipeline run 9
```

**Optimization for 20 Hours:**
To ensure the session lasts approximately 20 hours on a single H100, the following adjustments were made:
- **Data Volume:** Increased `limit` to `1000` projects to target ~100k+ samples.
- **Context Length:** Set `max_seq_len: 32768` for long-context fine-tuning on H100.
- **Batch Size:** Total effective batch size of 64 (`per_device_train_batch_size: 4` * `gradient_accumulation_steps: 16`).
- **Duration:** With ~100k samples and 3 epochs, the training is estimated to run for ~18-22 hours depending on the final filtered dataset size.

If you need to extend the duration further, increase `num_train_epochs` or `gradient_accumulation_steps` in `pipeline.yaml`.

### 4. Monitoring
Use TensorBoard to monitor the training progress:
```bash
tensorboard --logdir manifests/ --port 6006
```
(You will need to map port 6006 in your RunPod settings to access the UI).

## Project Structure
- `pipeline/`: Core logic and CLI implementation.
- `pipeline.yaml`: Pipeline configuration.
- `raw/`, `clean/`, `sft/`, `quality/`, `dataset/`: Data processing stages.
- `manifests/`: Logs and execution manifests.
- `models/`: Trained model adapters.
