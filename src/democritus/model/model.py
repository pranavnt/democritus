import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False  # Added parameter for scaled RoPE
    rope_scaling: Optional[dict] = None  # Added parameter for RoPE scaling config

    max_batch_size: int = 32
    max_seq_len: int = 2048

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scale: Optional[float] = None,
):
    # Adjust dimension for rotary embeddings
    dim = dim // 2
    # Generate frequency ranges
    freqs = 1.0 / (theta ** (torch.arange(0, dim, dtype=torch.float32) / dim))
    # Generate position ranges, scaled if necessary
    if scale is not None:
        t = torch.arange(end, dtype=torch.float32) / scale
    else:
        t = torch.arange(end, dtype=torch.float32)

    # Compute frequency bands
    freqs = torch.outer(t, freqs)
    # Convert to complex numbers
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    # Return complex rotary embeddings
    return torch.complex(freqs_cos, freqs_sin)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Change cache initialization to None
        self.cache_k = None
        self.cache_v = None

        # Store max sequence length for proper cache initialization
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Only use KV cache during evaluation/inference
        if not self.training:
            # Initialize or reset caches if needed
            if self.cache_k is None or self.cache_k.size(0) != bsz:
                self.cache_k = torch.zeros(
                    (bsz, self.max_seq_len, self.n_local_kv_heads, self.head_dim),
                    device=x.device, dtype=x.dtype
                )
                self.cache_v = torch.zeros(
                    (bsz, self.max_seq_len, self.n_local_kv_heads, self.head_dim),
                    device=x.device, dtype=x.dtype
                )

            self.cache_k = self.cache_k.to(xq.device)
            self.cache_v = self.cache_v.to(xq.device)

            self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, :start_pos + seqlen]
            values = self.cache_v[:bsz, :start_pos + seqlen]
        else:
            # During training, just use the current keys and values
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Initialize layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # Output layers
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        # Compute RoPE frequencies
        scale = None
        if params.use_scaled_rope:
            scale = params.rope_scaling.get("factor", 1.0) if params.rope_scaling else 1.0

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            theta=params.rope_theta,
            scale=scale,
        )

        # Apply rotary embedding scaling if specified
        if params.use_scaled_rope and params.rope_scaling:
            scaling_type = params.rope_scaling.get("type", "linear")
            if scaling_type == "linear":
                self.freqs_cis = self.freqs_cis * scale
            elif scaling_type == "dynamic":
                base = params.rope_scaling.get("factor", 1.0)
                self.freqs_cis = self.freqs_cis * (base ** torch.arange(self.freqs_cis.shape[0], device=self.freqs_cis.device))

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output