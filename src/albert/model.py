import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
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


def create_attention_mask(attention_mask: torch.Tensor, window_size: int | None = None) -> torch.Tensor:
    # attention_mask -> (batch, sequence_length)
    sequence_length = attention_mask.shape[1]
    device = attention_mask.device

    # batch, heads, sequence_length, sequence_length
    allowed = attention_mask[:, None, None, :].bool()

    if window_size is not None:
        # local layer: query i sees key j only if |i - j| <= window // 2
        positions = torch.arange(sequence_length, device=device)
        distance = (positions[None, :] - positions[:, None]).abs()
        allowed = allowed & (distance <= window_size // 2)

    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=device)
    return allowed | diagonal


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_drop = nn.Dropout(config.attention_dropout)

        is_global = config.is_global_layer(layer_idx)
        theta = config.global_rope_theta if is_global else config.local_rope_theta
        self.rotary = RotaryEmbedding(self.head_dim, theta, config.max_position_embeddings)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape

        # create q, k and v vectors
        qkv = self.Wqkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))

        # RoPE
        cos, sin = self.rotary(position_ids)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # softmax(q·k / sqrt(head_dim) where mask is True) @ v, for each head
        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)

        # join the heads again: (batch, heads, seq_len, head_dim) -> (batch, seq_len, hidden_size)
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden_size)
        return self.out_drop(self.Wo(out))


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_idx: int):
        super().__init__()

        # layer 0 gets its input from the embeddings, which already end with a LayerNorm
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.attn = ModernBertAttention(config, layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)
        self.mlp = ModernBertMLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask, position_ids)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertModel(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()

        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(ModernBertEncoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch, seq_len)

        global_mask = create_attention_mask(attention_mask)
        local_mask = create_attention_mask(attention_mask, self.config.local_attention)

        x = self.embeddings(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            mask = global_mask if self.config.is_global_layer(layer_idx) else local_mask
            x = layer(x, mask, position_ids)

        return self.final_norm(x)


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(x)))


class ModernBertForMaskedLM(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()

        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # weight tying: decoder and token embedding share one (vocab_size, hidden_size) matrix
        self.decoder.weight = self.model.embeddings.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # (batch, seq_len) -> (batch, seq_len, vocab_size) logits
        hidden = self.model(input_ids, attention_mask)
        return self.decoder(self.head(hidden))

    @classmethod
    def from_pretrained(cls, repository_id: str) -> "ModernBertForMaskedLM":
        model = cls(ModernBertConfig.from_pretrained(repository_id))
        state_dict = cls._load_checkpoint(repository_id)

        state_dict.setdefault("decoder.weight", state_dict["model.embeddings.tok_embeddings.weight"])

        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    @classmethod
    def _load_checkpoint(cls, repository_id: str) -> dict[str, torch.Tensor]:
        files = list_repo_files(repository_id)
        if "model.safetensors" in files:
            return load_file(hf_hub_download(repository_id, "model.safetensors"))

        path = hf_hub_download(repository_id, "pytorch_model.bin")
        return torch.load(path, map_location="cpu", weights_only=True)
