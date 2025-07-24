# feature_transforms/pipeline.py

"""
Pipeline for transformations, separated out by numeric and categorical features

You can simply put name as a string for base usage or can use a dictionary to pass in kwargs

Example Usage:
pipe = ColumnPipeline(
    numeric = ['standardize', 
               {'name' : 'impute', 'strategy' : 'median'},
               {'name' : 'whiten', 'eps' : 1e-4}
            ],

    categorical = [
                    {'name' : 'impute', 'strategy' : 'most_frequent'},
                    'one_hot'
                ]
)

df = pipe.fit_transform(df)

"""

from __future__ import annotations

import pandas as pd
from typing import Sequence, Mapping, Dict, Any, Union

from utils.registry import get_transform
from ._base import BaseTransform

TransformSpec = Union[str, Mapping[str, Any]]

class ColumnPipeline:
    def __init__(self, *, numeric: Sequence[TransformSpec] | None = None,
                 categorical: Sequence[TransformSpec] | None = None) -> None:
        self.numeric_specs = list(numeric or [])
        self.categorical_specs = list(categorical or [])

    def _build_chain(self, specs: Sequence[TransformSpec]) -> list[BaseTransform]:
        chain: list[BaseTransform] = []
        for spec in specs:
            if isinstance(spec, str):
                name, kwargs = spec, {}
            elif isinstance(spec, Mapping):
                if 'name' not in spec:
                    raise ValueError("Dict transform must contain a 'name' key")
                name = spec['name']
                kwargs = {k : v for k, v in spec.items() if k != 'name'}
            else:
                raise TypeError(f'Transform specification must be str or dict, got {spec!r}')
            
            cls = get_transform(name)
            chain.append(cls(**kwargs))
        return chain
    
    def fit(self, df: pd.DataFrame) -> ColumnPipeline:
        # column partition by dtype
        self._num_cols = df.select_dtypes(exclude = ['object', 'category', 'string']).columns.tolist()
        self._cat_cols = [c for c in df.columns if c not in self._num_cols]

        # instantiate + fit chains
        self._num_chain = self._build_chain(self.numeric_specs)
        self._cat_chain = self._build_chain(self.categorical_specs)

        num_df = df[self._num_cols]
        if not num_df.empty:
            for tr in self._num_chain:
                num_df = tr.fit_transform(num_df)
        
        cat_df = df[self._cat_cols]
        if not cat_df.empty:
            for tr in self._cat_chain:
                cat_df = tr.fit_transform(cat_df)
        
        self.output_dim = num_df.shape[1] + cat_df.shape[1]
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, '_num_chain'):
            raise RuntimeError('ColumnPipeline: call fit() first')
        
        num_df = df[self._num_cols]
        if not num_df.empty:
            for tr in self._num_chain:
                num_df = tr.transform(num_df)
        
        cat_df = df[self._cat_cols]
        if not cat_df.empty:
            for tr in self._cat_chain:
                cat_df = tr.transform(cat_df)
        
        out = pd.concat([num_df, cat_df], axis = 1)
        return out.astype('float32')
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def __repr__(self) -> str:
        return (
            f'ColumnPipeline(numeric = {self.numeric_specs}, '
            f'categorical = {self.categorical_specs}, '
            f'output_dim = {getattr(self, 'output_dim', None)})'
        )
