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

import math

def _get_act(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == 'gelu':
        return nn.GELU
    elif name == 'relu':
        return nn.ReLU
    elif name == 'leakyrelu':
        return nn.LeakyReLU
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'swish':
        # Swish = x * sigmoid(x); PyTorch 1.7+ has SiLU, which is equivalent
        return nn.SiLU
    raise ValueError("Activation must be one of: 'gelu', 'relu', 'leakyrelu', 'tanh', 'sigmoid', 'swish'")

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
    Base MLP Encoder for feature vectors -> d-dim embeddings

    Parameters:
     - input_dim: int - size D of pre-processed feature vectors
     - output_dim: int - embedding dimension d
     - hidden_dims: list[int] - e.g. [256, 256, 256] (default)
     - activation: {'gelu', 'relu', 'leakyrelu', 'tanh', 'sigmoid', 'swish'} (default: 'gelu')
     - dropout: float in [0, 1) (default: 0.15)
     - batchnorm: bool (default: True)
     - skip_connections: bool - residual links when shapes allow (default: True)
    """
    def __init__(self, *, input_dim: int, output_dim: int, hidden_dims: Sequence[int] | None = None, 
                 activation: str = 'gelu', dropout: float = 0.15, batchnorm: bool = True,
                 skip_connections: bool = True) -> None:
        super().__init__()

        self.output_dim: int = output_dim
        hidden_dims = list(hidden_dims or [256, 256, 256])

        dims = [input_dim] + hidden_dims + [output_dim]
        act_cls = _get_act(activation)
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]

            fc = nn.Linear(in_d, out_d, bias = False)

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

        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, orig_shape[-1])

        x = self.net(x)

        if len(orig_shape) > 2:
            x = x.view(*orig_shape[:-1], self.output_dim)

        return x

# --- Advanced RealMLP Implementation ---

try:
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier, RealMLP_TD_Regressor
    PYTABKIT_AVAILABLE = True
except ImportError:
    PYTABKIT_AVAILABLE = False

class ScalingLayer(nn.Module):
    """Learnable diagonal scaling layer (soft feature selection)."""
    def __init__(self, dim, lr_mult=6.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.lr_mult = lr_mult
    def forward(self, x):
        return x * self.scale

class NTKLinear(nn.Linear):
    """Linear layer with NTK parameterization (weights scaled by 1/sqrt(fan_in))."""
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight /= math.sqrt(self.in_features)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

class ParametricSELU(nn.Module):
    """SELU-α with learnable α in [0,1]."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)  # constrain to [0,1]
        return alpha * nn.functional.selu(x) + (1 - alpha) * x

class ParametricMish(nn.Module):
    """Mish-α with learnable α in [0,1]."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        mish = x * torch.tanh(nn.functional.softplus(x))
        return alpha * mish + (1 - alpha) * x

class FlatCosineScheduler:
    """Flat-cosine schedule for dropout/weight decay."""
    def __init__(self, base, total_steps):
        self.base = base
        self.total_steps = total_steps
    def __call__(self, step):
        # ramps from base to 0 as step increases
        return self.base * 0.5 * (1 + math.cos(math.pi * min(step, self.total_steps) / self.total_steps))
    
@register_encoder('RealMLP')
class RealMLP(nn.Module):
    """
    RealMLP: Meta-tuned MLP with advanced tricks (see NeurIPS 2024, 2407.04491)

    Options:
        - input_dim: int
        - output_dim: int
        - num_layers: int (default 3)
        - dim_layers: int (default 256)
        - activation: str (gelu [default], relu, leakyrelu, tanh, swish, selu, mish)
        - parametric_activation: bool (default False, True for RealMLP tricks)
        - skip_connections: bool (default True)
        - batchnorm: bool (default True)
        - dropout: float (default 0.15)
        - dropout_schedule: str ("flatcos" or None)
        - scaling_layer: bool (default True, RealMLP trick)
        - ntk_param: bool (default True, RealMLP trick)
        - output_clip: bool (default False for classif, True for regression)
        - task: str ("classification" or "regression")
        - optimizer: str ("adamw"), lr, weight_decay, betas, etc.
    """
    def __init__(self, *, input_dim, output_dim, num_layers=3, dim_layers=256, activation='gelu',
                 parametric_activation=False, skip_connections=True, batchnorm=True, dropout=0.15,
                 dropout_schedule=None, scaling_layer=True, ntk_param=True, output_clip=False,
                 task='classification'):
        super().__init__()
        import math
        self.output_dim = output_dim
        self.task = task
        dims = [input_dim] + [dim_layers] * num_layers + [output_dim]
        layers = []
        act = activation.lower()
        # Scaling layer (RealMLP trick)
        if scaling_layer:
            layers.append(ScalingLayer(input_dim))
        for i in range(num_layers):
            in_d, out_d = dims[i], dims[i+1]
            # NTK parameterization
            Linear = NTKLinear if ntk_param else nn.Linear
            fc = Linear(in_d, out_d)
            # Last layer: no BN/activation/dropout
            if i == num_layers - 1:
                layers.append(fc)
                break
            bn = nn.BatchNorm1d(out_d) if batchnorm else nn.Identity()
            # Parametric activations (RealMLP trick)
            if parametric_activation:
                if task == 'classification' and act == 'selu':
                    act_layer = ParametricSELU(out_d)
                elif task == 'regression' and act == 'mish':
                    act_layer = ParametricMish(out_d)
                else:
                    act_layer = _get_act(act)()
            else:
                act_layer = _get_act(act)()
            do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            block = nn.Sequential(fc, bn, act_layer, do)
            if skip_connections:
                proj = None if in_d == out_d else nn.Linear(in_d, out_d, bias=False)
                block = _Residual(block, proj)
            layers.append(block)
        self.net = nn.Sequential(*layers)
        self.output_clip = output_clip
        self.clip_min = None
        self.clip_max = None
    def forward(self, x):
        x = self.net(x)
        if self.output_clip and self.clip_min is not None and self.clip_max is not None:
            x = torch.clamp(x, self.clip_min, self.clip_max)
        return x
    def set_output_clip(self, y_train):
        # For regression: set min/max for output clipping
        self.clip_min = float(np.min(y_train))
        self.clip_max = float(np.max(y_train))

# --- Factory for RealMLP or pytabkit RealMLP ---
def get_realmlp(*, input_dim, output_dim, use_pytabkit=False, **kwargs):
    """Factory: returns RealMLP (native) or pytabkit RealMLP if available and requested."""
    if use_pytabkit and PYTABKIT_AVAILABLE:
        # Example: for classification
        return RealMLP_TD_Classifier(input_dim=input_dim, output_dim=output_dim, **kwargs)
    return RealMLP(input_dim=input_dim, output_dim=output_dim, **kwargs)
