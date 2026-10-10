import dataclasses
import json
from dataclasses import dataclass

from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ModernBertConfig:
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    global_attn_every_n_layers: int = 3
    local_attention: int = 128  # total window: each token sees +-64 tokens
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    max_position_embeddings: int = 8192
    norm_eps: float = 1e-5
    pad_token_id: int = 50283
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def is_global_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.global_attn_every_n_layers == 0

    @classmethod
    def from_dict(cls, raw: dict) -> "ModernBertConfig":

        configuration = dict(raw)

        rope_config = configuration.get("rope_parameters", {})
        if "full_attention" in rope_config:
            configuration.setdefault("global_rope_theta", rope_config["full_attention"]["rope_theta"])
        if "sliding_attention" in rope_config:
            configuration.setdefault("local_rope_theta", rope_config["sliding_attention"]["rope_theta"])

        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in configuration.items() if key in names})

    @classmethod
    def tiny(cls) -> "ModernBertConfig":
        return cls(hidden_size=256, intermediate_size=384, num_hidden_layers=7, num_attention_heads=4)

    @classmethod
    def from_pretrained(cls, repository_id: str) -> "ModernBertConfig":
        path = hf_hub_download(repository_id, "config.json")
        with open(path) as file:
            return cls.from_dict(json.load(file))
