from typing import overload
import torch
from torch import nn

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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        E = torch.empty(num_embeddings, embedding_dim)
        _ = nn.init.trunc_normal_(tensor=E, mean=0, std=1, a=-3, b=3)
        self.E: nn.Parameter = nn.Parameter(E, requires_grad=True)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.E[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device=None, dtype=None
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        gain = torch.ones(size=(d_model,))
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_vec = torch.sqrt(
            x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )
        result = x / rms_vec * self.gain
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        if not d_ffn:
            d_ffn = int(8 / 3 * d_model)
        self.w1: Linear = Linear(d_model, d_ffn)
        self.w2: Linear = Linear(d_ffn, d_model)
        self.w3: Linear = Linear(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: torch.Tensor, apply_dim: int):
    scaled = x - x.max(dim=apply_dim, keepdim=True).values.reshape(-1, 1)
    return torch.exp(scaled) / torch.exp(scaled).sum(
        dim=apply_dim, keepdim=True
    )

