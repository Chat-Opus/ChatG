#!/usr/bin/env python3
"""Build a CPU-only CSV question/answer memory and export an HF custom model."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel

from configuration_csv_memory import CsvMemoryConfig
from memory_logic import MemoryEngine, build_records_from_csv
from modeling_csv_memory import CsvMemoryModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train/export a deterministic CSV memory model (CPU only).")
    parser.add_argument("--csv", required=True, help="CSV with Question and Correct Answer columns")
    parser.add_argument("--output-dir", default="./local-model", help="Export directory")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads used by PyTorch")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.torch_threads))
    torch.manual_seed(42)

    records, source_stats = build_records_from_csv(csv_path)
    engine = MemoryEngine(records)

    # Register custom classes so save_pretrained writes the HF auto_map metadata.
    CsvMemoryConfig.register_for_auto_class()
    CsvMemoryModel.register_for_auto_class("AutoModel")

    config = CsvMemoryConfig(num_records=len(records), hidden_size=1)
    model = CsvMemoryModel(config).to("cpu")
    model.eval()

    # save_pretrained generates config.json + model.safetensors and copies custom code.
    model.save_pretrained(output_dir, safe_serialization=True)

    # Make the custom model code and lookup engine explicitly present in the export.
    here = Path(__file__).resolve().parent
    for filename in ("configuration_csv_memory.py", "modeling_csv_memory.py", "memory_logic.py"):
        shutil.copy2(here / filename, output_dir / filename)

    mapping_obj = engine.to_json_obj(source_stats=source_stats)
    (output_dir / "mapping.json").write_text(
        json.dumps(mapping_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "model_type": "csv-memory-lookup",
        "runtime_device": "cpu",
        "source_csv_name": csv_path.name,
        **source_stats,
        "exported_files": [
            "config.json", "model.safetensors", "mapping.json",
            "configuration_csv_memory.py", "modeling_csv_memory.py", "memory_logic.py",
        ],
        "note": "Rule-based lookup: no gradient-based neural training or general reasoning occurs.",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Verify that the exported HF custom model can be loaded and that its mapping works.
    reloaded = AutoModel.from_pretrained(
        str(output_dir), trust_remote_code=True, local_files_only=True
    ).to("cpu")
    if len(reloaded.memory.records) != len(records):
        raise RuntimeError("Export verification failed: mapping record count changed after reload")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "num_records": len(records),
        **source_stats,
        "load_test": "passed (AutoModel.from_pretrained with trust_remote_code=True)",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
