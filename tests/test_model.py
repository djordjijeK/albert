import torch
import torch.nn.functional as F

from albert.config import ModernBertConfig
from albert.model import ModernBertEmbeddings, ModernBertMLP, RotaryEmbedding, apply_rope

CFG = ModernBertConfig(vocab_size=16, hidden_size=8, intermediate_size=32, pad_token_id=0)


def test_parameter_names_match_the_checkpoint_and_have_no_bias():
    assert sorted(ModernBertEmbeddings(CFG).state_dict()) == ["norm.weight", "tok_embeddings.weight"]
    assert sorted(ModernBertMLP(CFG).state_dict()) == ["Wi.weight", "Wo.weight"]


def test_embeddings_give_one_normalized_vector_per_token():
    ids = torch.randint(1, CFG.vocab_size, (2, 5))  # skip id 0: the padding row starts at zero
    out = ModernBertEmbeddings(CFG)(ids)

    assert out.shape == (2, 5, CFG.hidden_size)
    torch.testing.assert_close(out.mean(-1), torch.zeros(2, 5), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.std(-1, correction=0), torch.ones(2, 5), atol=1e-3, rtol=0)


def test_mlp_is_gelu_of_first_half_times_second_half():
    mlp = ModernBertMLP(CFG).eval()
    x = torch.randn(2, 5, CFG.hidden_size)
    value, gate = (x @ mlp.Wi.weight.T).chunk(2, dim=-1)
    expected = (F.gelu(value) * gate) @ mlp.Wo.weight.T

    torch.testing.assert_close(mlp(x), expected)


def test_rope_gives_one_cos_and_sin_row_per_token_and_saves_no_weights():
    rope = RotaryEmbedding(head_dim=16, theta=10_000.0, max_seq_len=64)
    positions = torch.arange(10).expand(2, 10)

    cos, sin = rope(positions)

    assert cos.shape == sin.shape == (2, 10, 16)
    assert len(rope.state_dict()) == 0


def test_rope_does_not_change_vector_lengths():
    rope = RotaryEmbedding(head_dim=16, theta=10_000.0, max_seq_len=64)
    x = torch.randn(2, 4, 10, 16)
    cos, sin = rope(torch.arange(10).expand(2, 10))

    turned = apply_rope(x, cos, sin)

    torch.testing.assert_close(turned.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(turned[:, :, 0], x[:, :, 0])


def test_rope_scores_depend_only_on_the_distance_between_tokens():
    rope = RotaryEmbedding(head_dim=16, theta=10_000.0, max_seq_len=64)
    q = torch.randn(16).expand(1, 1, 10, 16)
    k = torch.randn(16).expand(1, 1, 10, 16)
    cos, sin = rope(torch.arange(10).expand(1, 10))

    scores = apply_rope(q, cos, sin)[0, 0] @ apply_rope(k, cos, sin)[0, 0].T

    torch.testing.assert_close(scores.diagonal(2), scores[0, 2].expand(8), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(scores[0, 2], scores[0, 3])
