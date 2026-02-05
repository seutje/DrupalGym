import json
import random
from pathlib import Path
from .manifest import Manifest, calculate_hash
from .logger import PipelineLogger

def run_packaging_stage(config: dict, logger: PipelineLogger, root: Path):
    passed_file = root / "quality" / "passed.jsonl"
    if not passed_file.exists():
        logger.error("quality/passed.jsonl not found.")
        return 1

    dataset_root = root / "dataset"
    version = "v1" # We could make this dynamic
    v_dir = dataset_root / version
    v_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = Manifest("dataset_packaging", v_dir)
    manifest.add_input("passed_sft", "1.0", calculate_hash(passed_file))

    # Load samples
    with open(passed_file, 'r', encoding='utf-8') as f:
        samples = [line for line in f]
    
    # Shuffle with fixed seed
    random.seed(config.get("seed", 42))
    random.shuffle(samples)
    
    total = len(samples)
    targets = config.get("dataset", {}).get("targets", {"train": 0.8, "valid": 0.1, "test": 0.1})
    
    train_end = int(total * targets.get("train", 0.8))
    valid_end = train_end + int(total * targets.get("valid", 0.1))
    
    splits = {
        "train": samples[:train_end],
        "valid": samples[train_end:valid_end],
        "test": samples[valid_end:]
    }
    
    for name, split_samples in splits.items():
        out_path = v_dir / f"{name}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for sample in split_samples:
                f.write(sample)
        manifest.add_output(f"{name}_split", f"dataset/{version}/{name}.jsonl", calculate_hash(out_path))

    stats = {split: len(data) for split, data in splits.items()}
    stats["total"] = total
    manifest.set_metrics(stats)
    
    manifest.save()
    
    # Update global index
    index_path = dataset_root / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path, 'r') as f:
            index = json.load(f)
    
    index[version] = {
        "timestamp": manifest.data["timestamp"],
        "metrics": stats,
        "path": f"dataset/{version}"
    }
    
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    logger.info(f"Dataset packaging complete. Version {version}. Stats: {stats}")
    return 0
