# feature_transforms/whiten.py

"""
ZCA whitening

Parameters:
eps - float - added to eigenvalues for numeric stability
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from utils.registry import register_transform
from ._base import BaseTransform

@register_transform('whiten')
class Whitening(BaseTransform):
    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.whitening_mat_: np.ndarray | None = None
    
    def fit(self, X: pd.DataFrame) -> Whitening:
        arr = X.to_numpy(dtype = float)
        
        # zero mean
        self.mean_ = np.mean(arr, axis = 0)
        Xc = arr - self.mean_

        # cov matrix
        cov = np.cov(Xc, rowvar = False)

        # eigen-decomp
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # add jitter to diagonal entry and retry
            jitter = 1e-6 * np.trace(cov) / cov.shape[0]
            eigvals, eigvecs = np.linalg.eigh(cov + np.eye(cov.shape[0]) * jitter)

        # build whitening mat
        inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + self.eps))
        self.whitening_mat_ = eigvecs @ inv_sqrt @ eigvecs.T

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.whitening_mat_ is None:
            raise RuntimeError('Whitening: must fit before transform')
        
        arr = X.to_numpy(dtype = float)
        Xc = arr - self.mean_
        Xw = Xc @ self.whitening_mat_

        return pd.DataFrame(Xw, index = X.index, columns = X.columns)

