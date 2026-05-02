import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        w = torch.empty(out_features, in_features)
        var: float = 2 / (in_features + out_features)
        std: float = var**0.5
        _ = nn.init.trunc_normal_(tensor=w, std=std, mean=0, a=-3 * std, b=3 * std)
        self.w: nn.Parameter = nn.Parameter(data=w, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        E = torch.empty(num_embeddings, embedding_dim)
        _ = nn.init.trunc_normal_(tensor=E, mean=0, std=1, a=-3, b=3)
        self.E: nn.Parameter = nn.Parameter(E, requires_grad=True)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.E[token_ids]


class RMSNorm(nn.Module):
    gain: nn.Parameter
    d_model: int
    eps: float

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.ones(size=(d_model,))
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_vec = torch.sqrt(x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps)
        result = x / rms_vec * self.gain
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if not d_ffn:
            d_ffn = int(8 / 3 * d_model)
        self.w1: Linear = Linear(d_model, d_ffn)
        self.w2: Linear = Linear(d_ffn, d_model)
        self.w3: Linear = Linear(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.w2(silu(self.w1(x)) * self.w3(x))
        return res


def softmax(x: torch.Tensor, apply_dim: int):
    scaled = x - x.max(dim=apply_dim, keepdim=True).values.reshape(-1, 1)
    return torch.exp(scaled) / torch.exp(scaled).sum(dim=apply_dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> torch.Tensor:
    d_k: int = Q.shape[-1]
    scores: torch.Tensor = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / (d_k**0.5)

    if mask is not None:
        scores.masked_fill_(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = einsum(attn_weights, V, "... q k, ... k d_v -> ... q d_v")
    return output


# class CausalMultiHeadSelfAttention(nn.Module):
#     """Multi-Head Self-Attention

#     This function implements section 3.2.2 of the Transformer paper. In particular,
#     given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
#     it to create queries, keys, and values, and then perform causal multi-headed attention with
#     those queries, keys, and values.

#     Args:
#         d_model: int
#             The dimensionality of the model embeddings and sublayer outputs.
#         num_heads: int
#             Number of heads to use in multi-headed attention. `d_model` must be
#             evenly divisible by `num_heads`.
#         positional_encoder: RotaryEmbedding
#             The RoPE module to use.

#     Returns:
#         Tensor of shape `(batch_size, sequence_length, d_model)`.
#     """

#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         positional_encoder: RotaryEmbedding,
#     ):
#         super().__init__()
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads

#         self.d_k = d_model // num_heads
#         self.d_v = self.d_k

#         self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
#         self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
#         self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

#         self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

#         self.positional_encoder = positional_encoder  # RoPE

#     def forward(
#         self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
#     ) -> Float[Tensor, " ... seq d_v"]:
#         """
#         Args:
#             x: The input to perform multi-headed self-attention on.
#             positional_ids: The positional indices along the sequence dimension of the input embeddings.

#         Returns:
#             Self-attention outputs.
#         """
#         *b, sequence_length, d_model = x.size()
#         assert d_model == self.d_model

#         Q = self.q_proj(x)
#         K = self.k_proj(x)
#         V = self.v_proj(x)

#         # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
#         Q, K, V = (
#             rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
#             for X in (Q, K, V)
#         )  # fmt: skip

#         if token_positions is None:
#             token_positions = einx.rearrange(
#                 "seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b)
#             )

#         # Duplicate token positions for each head
#         token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

#         Q = self.positional_encoder(Q, token_positions)
#         K = self.positional_encoder(K, token_positions)

#         # Construct causal mask
#         seq = torch.arange(sequence_length, device=x.device)
#         qi = einx.rearrange("query -> b... 1 query 1", seq, b=[1] * len(b))
#         kj = einx.rearrange("key   -> b... 1 1   key", seq, b=[1] * len(b))
#         causal_mask = qi >= kj  # (query, key)

#         # Shape: (..., num_heads, sequence_length, d_k)
#         attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

#         # Concatenate the attention output from all heads.
#         # (..., sequence_length, num_heads * d_v).
#         attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

#         # Apply the output projection
#         output = self.output_proj(attn_output)
#         return output


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
