from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        out_dim: int,
        activation_function: str = "tanh",
    ) -> None:
        super().__init__()

        self.activation_function = activation_function

        dims = (in_dim,) + hidden_dims + (out_dim,)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.aux_heads = nn.ModuleList(
            [nn.Linear(hidden_dims[i], out_dim) for i in range(len(hidden_dims))]
        )
        self.phago: Optional[dict[int, Any]] = None

    def attach_phagocytosis(self, phago: dict[int, Any]) -> None:
        self.phago = phago

    def detach_phagocytosis(self) -> None:
        self.phago = None

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_function == "tanh":
            return torch.tanh(x)
        if self.activation_function == "relu":
            return F.relu(x)
        if self.activation_function == "sigmoid":
            return torch.sigmoid(x)
        if self.activation_function == "identity":
            return x
        raise ValueError(f"Unknown activation function: {self.activation_function}")

    def _linear_with_optional_mask(
        self,
        li: int,
        layer: nn.Linear,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.phago is None or li not in self.phago:
            return layer(x)

        masked_weight = self.phago[li].masked_weight(layer.weight)
        return F.linear(x, masked_weight, layer.bias)

    def forward(self, x, return_acts: bool = False):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        r = [x]
        v = []

        h = x
        for li, layer in enumerate(self.layers):
            pre = self._linear_with_optional_mask(li, layer, h)
            v.append(pre)

            if li < len(self.layers) - 1:
                h = self._phi(pre)
                r.append(h)
            else:
                logits = pre
                r.append(logits)

        if return_acts:
            return logits, {"r": r, "v": v}
        return logits
