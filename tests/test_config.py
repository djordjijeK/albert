from albert.config import ModernBertConfig


def test_head_dim_is_hidden_size_divided_by_number_of_heads():
    assert ModernBertConfig(hidden_size=256, num_attention_heads=4).head_dim == 64


def test_every_third_layer_is_global_starting_at_layer_zero():
    cfg = ModernBertConfig(num_hidden_layers=7, global_attn_every_n_layers=3)
    assert [i for i in range(10) if cfg.is_global_layer(i)] == [0, 3, 6, 9]


def test_from_dict_reads_old_and_new_rope_formats_and_ignores_unknown_keys():
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
        cfg = ModernBertConfig.from_dict(raw)
        assert (cfg.hidden_size, cfg.global_rope_theta, cfg.local_rope_theta) == (256, 111.0, 222.0)


def test_from_dict_keeps_defaults_for_missing_keys():
    assert ModernBertConfig.from_dict({}) == ModernBertConfig()


def test_modernbert_base_config_on_the_hub_equals_our_defaults():
    assert ModernBertConfig.from_pretrained("answerdotai/ModernBERT-base") == ModernBertConfig()


def test_tiny_has_the_same_shape_as_ettin_17m():
    ettin, tiny = ModernBertConfig.from_pretrained("jhu-clsp/ettin-encoder-17m"), ModernBertConfig.tiny()
    for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"):
        assert getattr(ettin, name) == getattr(tiny, name), name
