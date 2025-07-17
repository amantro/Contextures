# adapters/kernel_to_multi.py

"""
Given a kernel object (k_obj) exposing '.compute(X1, X2)' -> (B, N) similarity
      and reference sets X_ref, A_ref representing {a1, ..., aN}

the adapter turns those similarities into sampling weights and draws 'n_samples'
independent a's for each query x

Output:
A: Tensor (B, n_samples, D) - sampled positive contexts
idx: LongTensor (B, n_samples) - indices into reference set
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from typing import Tuple

class KernelToMulti:
    def __init__(self, k_obj, *, n_samples: int = 1, temperature: float = 1.0, max_reference: int | None = None) -> None:
        """
        Parameters
         - k_obj: any object with 'compute(X1, X2) -> similarity'
         - n_samples: # of a's to draw per x (>= 1)
         - temperature: rescales logits sim / T (T < 1 -> sharper)
         - max_reference: subsample size of reference set when computing kernel (None = use all rows)
        """
        if n_samples < 1:
            raise ValueError('n_samples must be >= 1')
        self.k = k_obj
        self.n_samples = n_samples
        self.T = temperature
        self.max_reference = max_reference

        self._X_ref: Tensor | None = None
        self._A_ref: Tensor | None = None
    
    def fit(self, X_ref: Tensor, A_ref: Tensor) -> KernelToMulti:
        if X_ref.shape[0] != A_ref.shape[0]:
            raise ValueError('X_ref and A_ref must have same length')
        
        self._X_ref = X_ref.detach()
        self._A_ref = A_ref.detach()
        return self
    
    @torch.no_grad
    def transform(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
         - X: Tensor (B, D_x)
        
        Returns
         - A: Tensor (B, n_samples, D_a)
         - idxs: LongTensor (B, n_samples)
        """
        if self._X_ref is None:
            raise RuntimeError('Call .fit() before .transform()')
        
        # Optionally subsample reference pool for efficiency
        X_ref = self._X_ref
        A_ref = self._A_ref
        if self.max_reference and self.max_reference < X_ref.size(0):
            rows = np.random.choice(X_ref.size(0), self.max_reference, replace = False)
            X_ref = X_ref[rows]
            A_ref = A_ref[rows]
        
        # similarities + softmax
        sim = self.k.compute(X.float(), X_ref.float()) / self.T
        probs = torch.softmax(sim, dim = 1)

        # sample indices for each row
        idx = torch.multinomial(probs, num_samples = self.n_samples, replacement = True)
        A = A_ref[idx]

        return A, idx
