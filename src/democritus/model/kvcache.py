import jax
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class KVCache:
    k: jnp.ndarray  # shape: [n_layer, batch_size, seq_len, n_head, head_dim]
    v: jnp.ndarray  # shape: [n_layer, batch_size, seq_len, n_head, head_dim]

    def update(self, xk, xv, layer_idx: int, cur_pos: int, n_rep: int):
        """
        Update KV cache with new key-value pairs at the current position.

        Args:
            xk: New keys to add [batch_size, seq_len, n_head, head_dim]
            xv: New values to add [batch_size, seq_len, n_head, head_dim]
            layer_idx: Current layer index
            cur_pos: Current position in sequence
            n_rep: Number of repetitions (usually 1 during inference)
        """
        # Update cache at the correct position
        ck = jax.lax.dynamic_update_slice(
            self.k,
            jnp.bfloat16(xk[None, ...]),
            (layer_idx, 0, cur_pos, 0, 0)
        )
        cv = jax.lax.dynamic_update_slice(
            self.v,
            jnp.bfloat16(xv[None, ...]),
            (layer_idx, 0, cur_pos, 0, 0)
        )

        # Get all cached keys/values up to current position + new tokens
        seq_len = cur_pos + xk.shape[1]
        keys = ck[layer_idx, :, :seq_len]
        values = cv[layer_idx, :, :seq_len]

        return keys, values, KVCache(k=ck, v=cv)

    @classmethod
    def create(cls, batch_size: int, seq_len: int, n_layer: int, n_head: int, head_dim: int):
        """Create an empty KV cache"""
        k = jnp.zeros((n_layer, batch_size, seq_len, n_head, head_dim), dtype=jnp.bfloat16)
        v = jnp.zeros((n_layer, batch_size, seq_len, n_head, head_dim), dtype=jnp.bfloat16)
        return cls(k=k, v=v)