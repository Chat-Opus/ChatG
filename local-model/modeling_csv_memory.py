"""A Hugging Face-loadable wrapper around a deterministic CSV memory lookup.

This is deliberately NOT a causal language model. Its answer logic lives in the
mapping records and is exposed through answer_text(); forward() exists only to
make the artifact loadable through Transformers' AutoModel API.
"""
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

try:  # Works both as a local module and from HF's dynamic custom-code loader.
    from .configuration_csv_memory import CsvMemoryConfig
    from .memory_logic import MemoryEngine
except ImportError:
    from configuration_csv_memory import CsvMemoryConfig
    from memory_logic import MemoryEngine


class CsvMemoryModel(PreTrainedModel):
    config_class = CsvMemoryConfig
    base_model_prefix = "csv_memory"
    main_input_name = "input_ids"

    def __init__(self, config: CsvMemoryConfig):
        super().__init__(config)
        # A tiny CPU tensor makes the wrapper a valid serializable HF model.
        # Predictions are driven by mapping.json, not by this parameter.
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.memory = MemoryEngine([])
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model_path = Path(str(pretrained_model_name_or_path))
        if model_path.is_dir():
            mapping_path = model_path / "mapping.json"
            if mapping_path.is_file():
                model.memory = MemoryEngine.from_json(mapping_path)
        return loaded

    def answer_text(self, prompt: str, choice_format: str = "answer") -> str:
        return self.memory.answer(prompt, choice_format=choice_format)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required for the placeholder HF forward() method")
        batch_size, seq_len = input_ids.shape[:2]
        hidden = self.anchor.view(1, 1, 1).expand(batch_size, seq_len, self.config.hidden_size)
        return BaseModelOutput(last_hidden_state=hidden)
