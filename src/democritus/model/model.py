from typing import List, NamedTuple, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
from jax.experimental import mesh_utils
from jax.experimental.pallas.ops.gpu.rms_norm import rms_norm as pl_rms_norm

from democritus.model.config import ModelParams
from democritus.model.kvcache import KVCache

# First define the weight classes needed by the model
class LayerWeights(NamedTuple):
    wq: jax.Array
    wk: jax.Array
    wv: jax.Array
    wo: jax.Array
    w1: jax.Array
    w2: jax.Array
    w3: jax.Array
    ffn_norm: jax.Array
    attention_norm: jax.Array


class XfmrWeights(NamedTuple):
    tok_embeddings: jax.Array
    norm: jax.Array
    output: jax.Array
    layer_weights: List[LayerWeights]

class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array

    @classmethod
    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
    def new(
        cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int
    ) -> "KVCache":
        return cls(
            k=jnp.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16
            ),
            v=jnp.zeros(
                (layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16
            ),
        )

    def update(
        self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int
    ):
        ck = jax.lax.dynamic_update_slice(
            self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0, 0)
        )
        cv = jax.lax.dynamic_update_slice(
            self.v, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0, 0)
        )
        keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
        values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

        return keys, values, KVCache(k=ck, v=cv)

shard = jax.lax.with_sharding_constraint

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
    x = shard(x, PS())
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))


def apply_rotary_emb(
    xq: jax.Array,
    xk: jax.Array,
    freqs_cis: jax.Array,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        xk: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        freqs_cis: Precomputed frequencies [seq_len, dim]
    """
    seqlen = xq.shape[1]

    # Reshape xq and xk to split channels
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.reshape(*xk.shape[:-1], -1, 2)

    # Split into even and odd channels
    xq_even = xq_r[..., 0]
    xq_odd = xq_r[..., 1]
    xk_even = xk_r[..., 0]
    xk_odd = xk_r[..., 1]

    # Reshape freqs for broadcasting
    freqs = freqs_cis[:seqlen]  # [seq_len, dim]
    freqs_cos = jnp.cos(freqs)[None, :, None, :]  # [1, seq, 1, dim]
    freqs_sin = jnp.sin(freqs)[None, :, None, :]  # [1, seq, 1, dim]

    # Apply rotation
    out_q = jnp.stack(
        [
            xq_even * freqs_cos - xq_odd * freqs_sin,
            xq_odd * freqs_cos + xq_even * freqs_sin,
        ],
        axis=-1,
    )
    out_k = jnp.stack(
        [
            xk_even * freqs_cos - xk_odd * freqs_sin,
            xk_odd * freqs_cos + xk_even * freqs_sin,
        ],
        axis=-1,
    )

    return (
        out_q.reshape(xq.shape).astype(dtype),
        out_k.reshape(xk.shape).astype(dtype)
    )

def attention(
    x: jax.Array,
    layer_weights: LayerWeights,
    model_params,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, KVCache]:
    """Multi-head attention with KV cache."""
    bsz, seqlen, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

    # Project query, key, value
    xq = jnp.einsum("bse,ehd->bshd", x, layer_weights.wq)  # [batch, seq, heads, dim]
    xk = jnp.einsum("bse,ehd->bshd", x, layer_weights.wk)  # [batch, seq, kv_heads, dim]
    xv = jnp.einsum("bse,ehd->bshd", x, layer_weights.wv)  # [batch, seq, kv_heads, dim]

    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # Update KV cache
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

    # Attention scores
    scores = jnp.einsum("bqhd,bkhd->bhqk", xq, keys) / jnp.sqrt(model_params.head_dim)
    scores = scores.astype(jnp.float32)  # Always do attention softmax at float32

    if attn_mask is not None:
        scores = scores.at[..., :attn_mask.shape[-1]].add(attn_mask)

    # Mask padding tokens
    mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_scores = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)

    # Softmax and attention
    probs = jax.nn.softmax(padded_scores, axis=-1).astype(x.dtype)
    output = jnp.einsum("bhqk,bkhd->bqhd", probs, values)

    # Reshape and project output
    output = output.reshape(bsz, seqlen, -1)
    output = jnp.dot(output, layer_weights.wo)

    return shard(output, PS()), kvcache, scores

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 500000.0
) -> jax.Array:
    """Precompute frequencies for rotary embeddings."""
    # Generate frequency bands
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    # Generate time/position indices
    t = jnp.arange(end, dtype=jnp.float32)

    # Outer product -> [seq, dim/2]
    emb = jnp.einsum('i,j->ij', t, freqs)

    return emb  # [seq_len, dim/2]

def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
    """Feed forward network with SwiGLU activation.

    Args:
        x: Input of shape [batch, seq_len, hidden_dim]
        layer_weights: Layer weights containing w1, w2, w3 matrices

    Returns:
        Output of shape [batch, seq_len, hidden_dim]
    """
    x = shard(x, PS())  # [batch, seq, hidden]

    # First projection and activation
    w1 = layer_weights.w1.T  # [hidden, ffn_dim]
    w3 = layer_weights.w3.T  # [hidden, ffn_dim]
    w2 = layer_weights.w2.T  # [ffn_dim, hidden]

    # Project to larger dimension and apply SwiGLU
    h1 = jax.nn.silu(shard(jnp.einsum('bsh,hf->bsf', x, w1), PS(None, None, "mp")))
    h3 = shard(jnp.einsum('bsh,hf->bsf', x, w3), PS(None, None, "mp"))
    h = h1 * h3  # SwiGLU activation

    # Project back to hidden dimension
    out = shard(jnp.einsum('bsf,fh->bsh', h, w2), PS())

    return out

def xfmr(
    xfmr_weights: XfmrWeights,
    model_params: ModelParams,
    tokens: jax.Array,
    cur_pos: int,
    freqs_cis: jax.Array,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, KVCache]:
    h = xfmr_weights.tok_embeddings[tokens]
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(
            norm_x,
            xfmr_weights.layer_weights[i],
            model_params,
            cur_pos,
            i,
            freqs_cis,
            kvcache,
            attn_mask=attn_mask,
        )
        h = h + h_attn
        h = h + feed_forward(
            rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm),
            xfmr_weights.layer_weights[i],
        )

    normed = rms_norm(h, xfmr_weights.norm)
    print(normed.shape)
    print(xfmr_weights.output.shape)
    logits = jnp.dot(normed, xfmr_weights.output)
    return logits, kvcache, scores

# Add weight loading code at the end
@dataclass
class WeightConfig:
    """Configuration for weight loading and sharding."""
    dp_dim: str = "dp"
    mp_dim: str = "mp"
    fsdp_dim: str = "fsdp"


def create_mesh(device_count: int) -> jax.sharding.Mesh:
    """Creates device mesh for distributed execution."""
    devices = jax.devices()
    mesh_shape = (device_count, 1)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(device_mesh, ("mp", "fsdp"))


def create_partition_spec(key):
    dp = "dp"
    mp = "mp"
    fsdp = "fsdp"
    if "norm" in key:
        return PS()
    if "rope.freqs" in key:
        return PS()
    elif "tok_embeddings" in key:
        return PS(fsdp, mp)
    elif "output" in key:
        return PS(fsdp, mp)
    elif "w2" in key or "wo" in key:
        return PS(mp, fsdp)
    else:
        return PS(fsdp, mp)


def load_weights(
    ckpt_dir: Path, model_params, weight_config: Optional[WeightConfig] = None
) -> Tuple[XfmrWeights, jax.sharding.Mesh]:
    """Load and shard model weights across devices."""
    weight_config = weight_config or WeightConfig()
    mesh = create_mesh(jax.device_count())

    w = {}
    layer_weights = []

    for file in ckpt_dir.glob("*.npy"):
        name = ".".join(str(file).split("/")[-1].split(".")[:-1])
        weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)
        partition_spec = create_partition_spec(name)
        sharding = NamedSharding(mesh, partition_spec)
        if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
            weight = weight.T
            if "wq" in name or "wk" in name or "wv" in name:
                weight = weight.reshape(
                    -1,
                    model_params.n_local_heads
                    if "wq" in name
                    else model_params.n_local_kv_heads,
                    model_params.head_dim,
                )
        w[name] = jax.device_put(weight, sharding)

    for i in range(model_params.n_layers):
        layer_weights.append(
            LayerWeights(
                wq=w[f"layers.{i}.attention.wq.weight"],
                wk=w[f"layers.{i}.attention.wk.weight"],
                wv=w[f"layers.{i}.attention.wv.weight"],
                wo=w[f"layers.{i}.attention.wo.weight"],
                w1=w[f"layers.{i}.feed_forward.w1.weight"],
                w2=w[f"layers.{i}.feed_forward.w2.weight"],
                w3=w[f"layers.{i}.feed_forward.w3.weight"],
                ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
                attention_norm=w[f"layers.{i}.attention_norm.weight"],
            )
        )

    xfmr_weights = XfmrWeights(
        tok_embeddings=w["tok_embeddings.weight"],
        norm=w["norm.weight"],
        output=w["output.weight"],
        layer_weights=layer_weights,
    )

    return xfmr_weights, mesh

def generate(params, input_ids, max_length):
    """
    Generate tokens using KV caching for efficient inference.
    """
    batch_size = input_ids.shape[0]
    cur_pos = 0

    # Initialize KV cache
    kvcache = KVCache.create(
        batch_size=batch_size,
        seq_len=max_length,
        n_layer=params.n_layer,
        n_head=params.n_head,
        head_dim=params.head_dim
    )

    # Initial forward pass with the entire prompt
    freqs_cis = precompute_freqs_cis(params.head_dim, max_length)
    mask = create_mask(input_ids.shape[1])

    logits, kvcache = transformer(
        params,
        input_ids,
        freqs_cis[:input_ids.shape[1]],
        mask,
        kvcache,
        cur_pos
    )

    cur_pos += input_ids.shape[1]
    next_token = jnp.argmax(logits[:, -1], axis=-1)

    output_ids = [next_token]

    # Generate tokens one at a time
    for i in range(max_length - input_ids.shape[1]):
        # Create mask for the current position
        mask = create_mask(cur_pos + 1)[-1:]  # Only need mask for new token

        logits, kvcache = transformer(
            params,
            next_token[:, None],
            freqs_cis[cur_pos:cur_pos+1],
            mask,
            kvcache,
            cur_pos
        )

        next_token = jnp.argmax(logits[:, -1], axis=-1)
        output_ids.append(next_token)
        cur_pos += 1

        if next_token[0] == params.eos_token_id:
            break

    return jnp.stack(output_ids, axis=1)
