import torch
from torch import nn

from albert.config import ModernBertConfig


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.drop = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) ints -> (batch, seq_len, hidden_size)
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()

        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(value) * gate))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, max_seq_len: int):
        super().__init__()

        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)

        positions = torch.arange(max_seq_len, dtype=torch.float)

        angles = torch.outer(positions, inv_freq)
        angles = torch.cat((angles, angles), dim=-1)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[position_ids], self.sin[position_ids]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin
