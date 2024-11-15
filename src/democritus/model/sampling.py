from functools import partial
from pathlib import Path
import json
import torch
import numpy as np
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS

from democritus.model.model import xfmr, precompute_freqs_cis
from democritus.model.tokenizer import Tokenizer
from democritus.model.config import MODEL_CONFIGS, create_model_params
from democritus.model.model import LayerWeights, XfmrWeights, KVCache

DEFAULT_MODEL_PATH = Path.home() / ".llama" / "checkpoints" / "Llama3.1-8B"

def torch_to_jax(tensor):
    """Convert torch tensor to jax array."""
    # Convert BFloat16 to Float32 before converting to numpy
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return jnp.array(tensor.cpu().numpy())

def load_model(ckpt_path: Path = DEFAULT_MODEL_PATH):
    """Load model weights and tokenizer from PyTorch checkpoint"""
    # Load model parameters
    with open(ckpt_path / "params.json", "r") as f:
        params = json.load(f)

    # Create config from params
    config = MODEL_CONFIGS["8B"]  # Base config
    model_params = create_model_params(config)

    # Load tokenizer
    tokenizer = Tokenizer(str(ckpt_path / "tokenizer.model"))

    # Load PyTorch weights
    if torch.cuda.is_available():
        state_dict = torch.load(ckpt_path / "consolidated.00.pth", weights_only=True)
    else:
        state_dict = torch.load(ckpt_path / "consolidated.00.pth", map_location="cpu", weights_only=True)


    # Convert to JAX arrays and create layer weights
    w = {}
    layer_weights = []

    # Create device mesh for parallel execution
    devices = jax.devices()
    mesh_shape = (jax.device_count(), 1)
    device_mesh = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, ("mp", "fsdp"))

    # Helper for sharding
    def shard(array, spec):
        return jax.device_put(array, jax.sharding.NamedSharding(mesh, spec))

    # Convert and organize weights
    for name, param in state_dict.items():
        if "layers" in name:
            layer_idx = int(name.split(".")[1])
            while len(layer_weights) <= layer_idx:
                layer_weights.append({})

            weight = torch_to_jax(param)

            # Handle attention weights
            if any(k in name for k in ["wq", "wk", "wv", "wo"]):
                weight = weight.T

            if "wq" in name:
                layer_weights[layer_idx]["wq"] = weight.reshape(-1, model_params.n_local_heads, model_params.head_dim)
            elif "wk" in name:
                layer_weights[layer_idx]["wk"] = weight.reshape(-1, model_params.n_local_kv_heads, model_params.head_dim)
            elif "wv" in name:
                layer_weights[layer_idx]["wv"] = weight.reshape(-1, model_params.n_local_kv_heads, model_params.head_dim)
            elif "wo" in name:
                layer_weights[layer_idx]["wo"] = weight
            elif "w1" in name:
                layer_weights[layer_idx]["w1"] = weight
            elif "w2" in name:
                layer_weights[layer_idx]["w2"] = weight
            elif "w3" in name:
                layer_weights[layer_idx]["w3"] = weight
            elif "attention_norm" in name:
                layer_weights[layer_idx]["attention_norm"] = weight
            elif "ffn_norm" in name:
                layer_weights[layer_idx]["ffn_norm"] = weight
        else:
            weight = torch_to_jax(param)
            if "tok_embeddings" in name:
                w["tok_embeddings"] = weight
            elif "norm" in name:
                w["norm"] = weight
            elif "output" in name:
                w["output"] = weight.T

    # Convert to NamedTuples
    layer_weights = [
        LayerWeights(
            wq=shard(l["wq"], PS("fsdp", "mp")),
            wk=shard(l["wk"], PS("fsdp", "mp")),
            wv=shard(l["wv"], PS("fsdp", "mp")),
            wo=shard(l["wo"], PS("mp", "fsdp")),
            w1=shard(l["w1"], PS("fsdp", "mp")),
            w2=shard(l["w2"], PS("mp", "fsdp")),
            w3=shard(l["w3"], PS("fsdp", "mp")),
            ffn_norm=shard(l["ffn_norm"], PS()),
            attention_norm=shard(l["attention_norm"], PS())
        )
        for l in layer_weights
    ]

    xfmr_weights = XfmrWeights(
        tok_embeddings=shard(w["tok_embeddings"], PS("fsdp", "mp")),
        norm=shard(w["norm"], PS()),
        output=shard(w["output"], PS("fsdp", "mp")),
        layer_weights=layer_weights
    )

    return xfmr_weights, model_params, tokenizer, mesh

@partial(jax.jit, static_argnames=('temperature', 'top_p'))
def nucleus_sampling(
    logits: jax.Array,
    temperature: float = 0.8,
    top_p: float = 0.9,
    key: Optional[jax.random.PRNGKey] = None,
) -> jax.Array:
    """Nucleus sampling implementation (also known as top-p)."""
    raise NotImplementedError

@partial(jax.jit)
def greedy_sampling(logits: jax.Array) -> jax.Array:
    """Greedy sampling implementation (always selects highest probability token)."""
    return jnp.argmax(logits, axis=-1)

def generate(
    xfmr_weights,
    model_params,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    sampling_method: str = "nucleus",
    key: Optional[jax.random.PRNGKey] = None
) -> str:
    """Generate text using the model.

    Args:
        xfmr_weights: Model weights
        model_params: Model parameters
        tokenizer: Tokenizer
        prompt: Prompt to generate text from
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for nucleus sampling
        top_p: Top-p for nucleus sampling
        sampling_method: Either "nucleus" or "greedy"
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize KV cache and RoPE
    kvcache = KVCache.new(
        model_params.n_layers,
        1,  # batch size
        model_params.max_seq_len,
        model_params.n_local_kv_heads,
        model_params.head_dim
    )

    freqs_cis = precompute_freqs_cis(
        model_params.head_dim,
        model_params.max_seq_len,
        model_params.rope_theta
    )

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")
    input_ids = jnp.array(input_ids)[None, :]  # Add batch dimension

    generated = []
    cur_pos = 0

    # Generation loop
    for i in range(max_tokens):
        key, subkey = jax.random.split(key)

        # Forward pass
        logits, kvcache, _ = xfmr(
            xfmr_weights,
            model_params,
            input_ids if i == 0 else next_token[:, None],
            cur_pos,
            freqs_cis,
            kvcache
        )

        # Sample next token based on method
        if sampling_method == "greedy":
            next_token = greedy_sampling(logits[:, -1, :])
        else:  # nucleus sampling
            next_token = nucleus_sampling(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                key=subkey
            )

        # Update position and check stop condition
        cur_pos += 1 if i > 0 else input_ids.shape[1]
        if next_token.item() in tokenizer.stop_tokens:
            break

        generated.append(next_token.item())

    return tokenizer.decode(generated)

if __name__ == "__main__":
    import jax

    print("Loading model...")
    xfmr_weights, model_params, tokenizer, mesh = load_model()

    prompt = """My favorite color is blue. I like it because"""

    print("\nGenerating with prompt:", prompt)
    print("\nGenerating...")

    with mesh:
        output = generate(
            xfmr_weights=xfmr_weights,
            model_params=model_params,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=20,
            temperature=0.8,
            top_p=0.9,
            sampling_method="greedy",
            key=jax.random.PRNGKey(0)
        )

    print(f"\nOutput:\n{output}")