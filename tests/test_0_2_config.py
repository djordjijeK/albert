"""Increment 0.2: the config."""

import pytest

from albert.config import ModernBertConfig

MODERNBERT_BASE = {
    "vocab_size": 50368,
    "hidden_size": 768,
    "intermediate_size": 1152,
    "num_hidden_layers": 22,
    "num_attention_heads": 12,
    "global_attn_every_n_layers": 3,
    "local_attention": 128,
    "global_rope_theta": 160_000.0,
    "local_rope_theta": 10_000.0,
    "max_position_embeddings": 8192,
    "norm_eps": 1e-5,
    "pad_token_id": 50283,
    "embedding_dropout": 0.0,
    "attention_dropout": 0.0,
    "mlp_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_cutoff_factor": 2.0,
}


def differences(cfg, expected):
    """{field: (actual, expected)} for each field that is wrong or missing."""
    actual = {name: getattr(cfg, name, "<missing>") for name in expected}
    return {name: (actual[name], value) for name, value in expected.items() if actual[name] != value}


def test_defaults_are_modernbert_base():
    assert differences(ModernBertConfig(), MODERNBERT_BASE) == {}


def test_head_dim_and_global_layers():
    base, tiny = ModernBertConfig(), ModernBertConfig.tiny()
    assert base.head_dim == tiny.head_dim == 64
    assert [i for i in range(base.num_hidden_layers) if base.is_global_layer(i)] == [0, 3, 6, 9, 12, 15, 18, 21]
    assert [i for i in range(tiny.num_hidden_layers) if tiny.is_global_layer(i)] == [0, 3, 6]


def test_from_dict_reads_old_and_new_rope_formats():
    old = {"model_type": "modernbert", "hidden_size": 256, "global_rope_theta": 111.0, "local_rope_theta": 222.0}
    new = {
        "architectures": ["ModernBertForMaskedLM"],
        "hidden_size": 256,
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 111.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 222.0},
        },
    }
    for raw in (old, new):
        cfg = ModernBertConfig.from_dict(raw)  # unknown keys are ignored
        assert (cfg.hidden_size, cfg.global_rope_theta, cfg.local_rope_theta) == (256, 111.0, 222.0)
        assert cfg.num_hidden_layers == 22  # missing keys keep their default


def test_from_dict_rejects_features_we_do_not_implement():
    for bad in ({"attention_bias": True}, {"mlp_bias": True}, {"norm_bias": True}, {"hidden_activation": "silu"}):
        with pytest.raises(ValueError):
            ModernBertConfig.from_dict(bad)
    ModernBertConfig.from_dict({"decoder_bias": True, "hidden_activation": "gelu"})  # ModernBERT has these: accept


def test_from_pretrained_reads_hub_configs():
    """Downloads two small config.json files (network on the first run, then cached)."""
    assert differences(ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base"), MODERNBERT_BASE) == {}

    ettin, tiny = ModernBertConfig.from_pretrained("jhu-clsp/ettin-encoder-17m"), ModernBertConfig.tiny()
    shape = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"]
    assert differences(ettin, {name: getattr(tiny, name) for name in shape}) == {}
    assert ettin.local_rope_theta == 160_000.0  # Ettin uses the same theta in all layers
