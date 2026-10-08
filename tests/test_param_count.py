from model.config import ModelConfig
from model.model import TinySQL


def test_full_model_param_count():
    model = TinySQL(ModelConfig())
    assert model.num_params() == 125_077_824 == ModelConfig().expected_params()


def test_untied_param_count():
    cfg = ModelConfig(tie_embeddings=False)
    assert TinySQL(cfg).num_params() == cfg.expected_params() == 125_077_824 + 32_768 * 576
