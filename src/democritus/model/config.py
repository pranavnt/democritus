from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim_multiplier: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    max_seq_len: int


MODEL_CONFIGS = {
    "8B": ModelConfig(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        max_seq_len=4096,
    ),
}


class ModelParams(NamedTuple):
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool


def create_model_params(config: ModelConfig) -> ModelParams:
    return ModelParams(
        n_layers=config.n_layers,
        n_local_heads=config.n_heads,
        n_local_kv_heads=config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        use_scaled_rope=config.use_scaled_rope,
    )
