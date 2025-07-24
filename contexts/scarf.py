from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import register_context
from utils.types import DataFrame
from contexts.base import Contexts

@register_context('scarf')
class SCARF(Contexts):
    """
    SCARF (Self-supervised Contrastive Learning using Random Feature Corruption) knowledge component.
    
    Implements feature corruption for self-supervised learning by randomly replacing
    features with values sampled from marginal distributions.
    """
    def __init__(self, num_context_samples: int, distribution: str = 'uniform', corruption_rate: float = 0.6):
        self.num_context_samples = num_context_samples
        self.distribution = distribution
        self.corruption_rate = corruption_rate
        self.uniform_eps = 1e-6
    
    def fit(self, dataset: DataFrame):
        df = dataset.copy()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category").cat.codes
        for col in df.select_dtypes(include="category").columns:
            df[col] = df[col].cat.codes

        num = df.select_dtypes(include=[np.number]).astype(np.float32)
        if num.empty:
            raise ValueError("no numeric columns left after conversion")

        self.context_dim = num.shape[1]

        low  = torch.tensor(num.min().values,  dtype=torch.float32)
        high = torch.tensor(num.max().values,  dtype=torch.float32)
        self.features_low, self.features_high = low, high

        eps = self.uniform_eps
        if self.distribution == "uniform":
            self.marginals = Uniform(low - eps, high + eps)

        elif self.distribution == "gaussian":
            mean = (high + low) / 2
            std  = (high - low) / 4
            std  = torch.where(std == 0, torch.full_like(std, eps), std)
            self.marginals = Normal(mean, std)

        elif self.distribution == "bimodal":
            std = (high - low) / 8
            std = torch.where(std == 0, torch.full_like(std, eps), std)
            self.marginals_low  = Normal(low,  std)
            self.marginals_high = Normal(high, std)

        else:
            raise NotImplementedError(f"Unsupported prior distribution: {self.distribution}")

        return self

    def _sample(self, x: Tensor) -> Tensor:
        """
        Apply SCARF corruption to input features.
        Args:
            x: Input tensor of shape (batch_size, num_features)
        Returns:
            corrupted_x: Tensor of shape (batch_size, num_context_samples, num_features)
        """
        batch_size, num_features = x.size()
        r = self.num_context_samples
        # Create corruption mask for all r contexts in parallel
        corruption_mask = (torch.rand(batch_size, r, num_features, device=x.device) < self.corruption_rate)
        # Sample random values for all r contexts in parallel
        if self.distribution in ['uniform', 'gaussian']:
            # marginals.sample((batch_size, r)) returns (batch_size, r, num_features)
            x_random = self.marginals.sample(torch.Size((batch_size, r))).to(x.device)
        elif self.distribution == 'bimodal':
            x_random_low = self.marginals_low.sample(torch.Size((batch_size, r))).to(x.device)
            x_random_high = self.marginals_high.sample(torch.Size((batch_size, r))).to(x.device)
            mode_choice = torch.rand(batch_size, r, device=x.device) > 0.5
            mode_choice = mode_choice.unsqueeze(-1).expand(-1, -1, num_features)
            x_random = torch.where(mode_choice, x_random_low, x_random_high)
        # Expand x to (batch_size, r, num_features)
        x_expanded = x.unsqueeze(1).expand(-1, r, -1)
        # Apply corruption mask
        x_corrupted = torch.where(corruption_mask, x_random, x_expanded)
        return x_corrupted
