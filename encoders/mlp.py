# encoders/mlp.py

"""
MLP that maps feature vectors -> d-dim embeddings

Notes: 
 - Default architecture is [64, 64] hidden units
 - GELU is used for non-linearity by default but can be changed with 'activation = 'relu''
 - BatchNorm, Dropout, skip connections ON by default
 - No BatchNorm, Dropout, or activation in output layer
 - 'output_dim = d' is required and becomes 'self.output_dim'
"""

from __future__ import annotations
from typing import Sequence, Callable, Optional

import pandas as pd
import numpy as np

import torch
from torch import nn

from utils.registry import register_encoder

def _get_act(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == 'gelu':
        return nn.GELU
    elif name == 'relu':
        return nn.ReLU
    return ValueError("Activation must be 'gelu' or 'relu'")

class _Residual(nn.Module):
    # Simple residual wrapper: y = F(x) + P(x)
    def __init__(self, fn: nn.Module, proj: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.fn = fn
        self.proj = proj or nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + self.proj(x)
    
@register_encoder('MLPEncoder')
class MLPEncoder(nn.Module):
    """
    Parameters:
     - input_dim: int - size D of pre-processed feature vectors
     - output_dim: int - embedding dimension d
     - hidden_dims: list[int] - e.g. [64, 64] (default)
     - activation: {'gelu', 'relu'}
     - dropout: float in [0, 1)
     - batchnorm: bool
     - skip_connections: bool - residual links when shapes allow

    """
    def __init__(self, *, input_dim: int, output_dim: int, hidden_dims: Sequence[int] | None = None, 
                 activation: str = 'gelu', dropout: float = 0.15, batchnorm: bool = True,
                 skip_connections: bool = True) -> None:
        super().__init__()

        self.output_dim: int = output_dim
        hidden_dims = list(hidden_dims or [64, 64])

        dims = [input_dim] + hidden_dims + [output_dim]
        act_cls = _get_act(activation)
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]

            fc = nn.Linear(in_d, out_d)

            # last layer: no BN / activation / dropout
            if i == len(dims) - 2:
                layers.append(fc)
                break

            bn = nn.BatchNorm1d(out_d) if batchnorm else nn.Identity()
            act = act_cls()
            do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            block = nn.Sequential(fc, bn, act, do)

            if skip_connections:
                proj = None if in_d == out_d else nn.Linear(in_d, out_d, bias = False)
                block = _Residual(block, proj)
            
            layers.append(block)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype = torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype = torch.float32)
        elif not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype = torch.float32)
        
        x = x.to(next(self.parameters()).device)
        return self.net(x)
