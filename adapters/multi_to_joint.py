# adapters/multi_to_joint.py

"""
Given samples A ~ p(a | x) for each x in a batch, build an empirical
joint distribution p'(a, a') by pairing *different* examples

Simplest way (unbiased):
 - Flatten all A's into one long list
 - Sample 2 *independent* items *with replacement* -> joint pair

Returned Tensors:
 - A1, A2: (B, n_pairs, D) such that each row forms an independent draw from product measure

You can pass 'coupled = True' if you want to avoid pairing an element
with itself when the flattened pool is small
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from typing import Tuple

class MultiToJoint:
    def __init__(self, *, n_pairs: int = 1, coupled: bool = False) -> None:
        if n_pairs < 1:
            raise ValueError('n_pairs must be >= 1')
        
        self.n_pairs = n_pairs
        self.coupled = coupled # avoid identical indices if possible

    @torch.no_grad
    def transform(self, A: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
         - A: Tensor (B, r, D) - multi samples per x
        
        Returns
         - A1, A2: Tensor (B, n_pairs, D) - joint complex pairs
        """
        B, r, D = A.shape
        pool = A.reshape(-1, D)
        N = pool.shape[0]

        device = A.device
        idx1 = torch.randint(0, N, (B, self.n_pairs), device = device)

        if not self.coupled:
            idx2 = torch.randint(0, N, (B, self.n_pairs), device = device)
            return pool[idx1], pool[idx2]
        
        # ensure pool[idx1] != pool[idx2] (coupled = True)
        idx2 = torch.randint(0, N, (B, self.n_pairs), device = device)
        same_mask = (pool[idx1] == pool[idx2]).all(dim = -1)
        while same_mask.any():
            resample = torch.randint(0, N, (same_mask.sum().item(), ), device = device)
            idx2[same_mask] = resample
            same_mask = (pool[idx1] == pool[idx2]).all(dim = -1)
        
        return pool[idx1], pool[idx2]