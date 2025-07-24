# -*- coding: utf-8 -*-
'''
Usage:
$ python run.py
$ python run.py config=my.yaml
'''

from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import yaml
from yaml import safe_load
from easydict import EasyDict as edict
from sklearn.preprocessing import LabelEncoder

from scripts.loader import load_dataset, load_openml_dataset
from feature_transforms.pipeline import ColumnPipeline
from utils.registry import get_encoder, get_loss, get_context
from trainer.trainer import SVDTrainer
from downstream import run_probe

from statistics import mean, stdev

# Helper functions
def load_cfg(path: str | Path) -> edict:
    with open(path, 'r') as f:
        return edict(yaml.safe_load(f))


def build_feature_pipeline(fp_cfg: Dict[str, Any]) -> ColumnPipeline:
    if fp_cfg.get('name', '').lower() == 'identity':
        return ColumnPipeline()
    return ColumnPipeline(numeric = fp_cfg.get('numeric', []),
                          categorical = fp_cfg.get('categorical', []))


def train_val_test_split(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = len(X)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.9 * n)
    tr, va, te = perm[:n_train], perm[n_train:n_val], perm[n_val:]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


# Main
def main(cfg: edict) -> None:
    torch.manual_seed(cfg["global"]["seed"])
    np.random.seed(cfg["global"]["seed"])
    device = torch.device(cfg["global"]["device"])

    results_dir = Path(cfg["global"]["results_dir"])
    (results_dir / "models").mkdir(parents = True, exist_ok = True)

    # Detect whether user asked for OpenML task:
    # dataset.openml_task_id(int) or dataset.tag starts with 'openml__<task_id>'
    openml_task_id = cfg.dataset.get('openml_task_id')
    if openml_task_id is None and cfg.dataset.get('tag', '').startswith('openml__'):
        openml_task_id = int(cfg.dataset.tag.split('__')[-1])
    
    use_openml = openml_task_id is not None

    # optional: run one split only (parallel grid search)
    split_cli_idx = int(getattr(cfg, '_split_cli_idx', -1))

    if use_openml:
        # one logical 'dataset'
        tags = [f'openml_task_{openml_task_id}']
        X_full, y_full, meta, split_list = load_openml_dataset(openml_task_id,
                                                               val_ratio = float(cfg.dataset.get('val_ratio', 0.1)),
                                                               random_state = cfg['global']['seed'])
        # choose subset of splits
        splits_to_run = [split_list[split_cli_idx]] if split_cli_idx >= 0 else split_list
    else:
        if 'tag' in cfg.dataset:
            tags = [cfg.dataset.tag]
        else:
            big_yaml = safe_load(open('configs/datasets.yaml'))
            tags = big_yaml[cfg.dataset.group]

    for tag in tags:
        print(f'\n--- {tag} ---')

        if use_openml:
            X_df, y, = X_full, y_full
        else:
            X_df, y, meta = load_dataset(tag)
            splits_to_run = [train_val_test_split(X_df.values, y)]

        if meta['target_type'] in ('classification', 'binary'):
            le = LabelEncoder()
            y = le.fit_transform(y)

        all_test_metrics = []
        
        # iterate over splits
        for split_id, (tr_idx, va_idx, te_idx) in enumerate(splits_to_run):
            print(f'   > split #{split_id}    '
                  f'| train = {len(tr_idx)} val = {len(va_idx)} test = {len(te_idx)}')
            
            # slice dataframes / array
            Xtr_df = X_df.iloc[tr_idx]
            Xva_df = X_df.iloc[va_idx]
            Xte_df = X_df.iloc[te_idx]
            ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

            # feature pipeline
            pipe = build_feature_pipeline(cfg.feature_preprocessing)
            Xtr = pipe.fit_transform(Xtr_df).to_numpy(dtype = np.float32)
            Xva = pipe.transform(Xva_df).to_numpy(dtype = np.float32) if Xva_df is not None else None
            Xte = pipe.transform(Xte_df).to_numpy(dtype = np.float32)

            # context
            CtxCls = get_context(cfg.context.name)
            context = CtxCls(**cfg.context.parameters)
            context.fit(Xtr_df)
            context_collate = context.get_collate_fn()

            # datasets / loaders
            train_bs = int(cfg['train']['batch_size'])
            def make_loader(X: np.ndarray, *, shuffle = False):
                x_t = torch.tensor(X, dtype = torch.float32, device = device)
                ds = TensorDataset(x_t)
                return DataLoader(ds, batch_size = train_bs, shuffle = shuffle, collate_fn = context_collate)
            
            train_loader = make_loader(Xtr, shuffle = True)

            # encoders
            EncCls   = get_encoder(cfg.encoder.name)
            x_enc = EncCls(input_dim = pipe.output_dim, **cfg.encoder.parameters).to(device)
            a_enc = EncCls(input_dim = context.context_dim, **cfg.encoder.parameters).to(device)

            # optimizer & loss
            params = list(x_enc.parameters()) + list(a_enc.parameters())
            optim = torch.optim.Adam(params, lr = float(cfg['train']['lr']))
            LossCls = get_loss(cfg.losses.name)
            criterion = LossCls(**cfg.losses.parameters)

            # trainer
            trainer = SVDTrainer(x_enc, a_enc, optim, criterion,
                                train_loader, cfg.train.num_epochs,
                                device = device)
            trainer.train()
            trainer.save_model(results_dir / 'models')

            # frozen probe
            for p in x_enc.parameters():
                p.requires_grad_(False)

            print('Extracting train/val/test features...')
            def feats(X):
                t = torch.tensor(X, dtype = torch.float32, device = device)
                with torch.no_grad():
                    return x_enc(t).cpu()
                
            f_tr, f_va, f_te = feats(Xtr), feats(Xva) if Xva is not None else None, feats(Xte)

            probe_kind = cfg['probe']['kind']
            probe_params = cfg['probe'].get('params', {})

            probe, probe_res = run_probe(
                kind        = probe_kind,
                features    = f_tr,
                targets     = ytr,
                task_type   = meta['target_type'],
                X_val       = f_va, y_val = yva if Xva is not None else None,
                X_test      = f_te, y_test = yte,
                **probe_params # defined in config
            )

            all_test_metrics.append(probe_res['test_metrics'])

    def _agg(v):
        return dict(mean=float(np.mean(v)),
                    stderr=float(np.std(v) / (len(v)**0.5) if len(v) > 1 else 0))

    summary = {k: _agg([m[k] for m in all_test_metrics])
                             for k in all_test_metrics[0]}
    
    print('Aggregated test metrics:', summary)

    suffix = 'openml' if use_openml else 'full'
    out_json = results_dir / f'{tag}_{suffix}.json'
    with open(out_json, 'w') as f:
        json.dump(dict(
            meta = meta,
            splits = all_test_metrics,
            agg = summary,
            config = cfg,
        ), f, indent = 2)

# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/config.yaml',
                        help='Path to YAML experiment file')
    parser.add_argument('--split', type=int, default=-1,
                        help='Run **only** this OpenML split index '
                             '(0-based).  -1 = all (default)')
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    cfg['_split_cli_idx'] = args.split
    main(cfg)