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

## Project Structure
- `pipeline/`: Core logic and CLI implementation.
- `pipeline.yaml`: Pipeline configuration.
- `raw/`, `clean/`, `sft/`, `quality/`, `dataset/`: Data processing stages.
- `manifests/`: Logs and execution manifests.
- `models/`: Trained model adapters.
