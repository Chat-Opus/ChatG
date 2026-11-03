"""Hugging Face configuration for the CSV memory lookup model."""
from transformers import PretrainedConfig


class CsvMemoryConfig(PretrainedConfig):
    model_type = "csv-memory-lookup"

    def __init__(self, num_records: int = 0, hidden_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_records = int(num_records)
        # Hidden size belongs only to the minimal HF-compatible wrapper.
        self.hidden_size = int(hidden_size)
        self.architectures = ["CsvMemoryModel"]
