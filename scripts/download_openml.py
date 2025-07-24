# scripts/download_openml.py

from __future__ import annotations
import os, sys, re, gzip, json, argparse, multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import numpy as np, yaml, openml
from tqdm import tqdm

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CFG  = ROOT / "configs" / "datasets.yaml"

def _tid(tag: str) -> int:
    m = re.match(r".*__(\d+)$", tag)
    if m is None:
        raise ValueError(tag)
    return int(m.group(1))

def _infer_target(y: np.ndarray) -> str:
    ya = np.asarray(y)
    if np.issubdtype(ya.dtype, np.floating):
        u = np.unique(ya[~np.isnan(ya)])
        if np.allclose(u, u.astype(int)) and u.size <= 20:
            return "binary" if u.size == 2 else "classification"
        return "regression"
    if np.issubdtype(ya.dtype, np.integer):
        u = np.unique(ya)
        return "binary" if u.size == 2 else "classification"
    u = np.unique(ya.astype(str))
    return "binary" if u.size == 2 else "classification"

def _download(tag: str) -> Tuple[str, bool, str]:
    out = DATA / tag
    try:
        if (out / "X.npy.gz").is_file() and (out / "y.npy.gz").is_file():
            return tag, True, "cached"
        task = openml.tasks.get_task(_tid(tag))
        Xdf, yser, cat_mask, _ = task.get_dataset().get_data(
            target=task.target_name, dataset_format="dataframe")
        X, y = Xdf.to_numpy(), yser.to_numpy()
        ttype = _infer_target(y)
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "X.npy.gz", "wb") as f:
            np.save(f, X)
        with gzip.open(out / "y.npy.gz", "wb") as f:
            np.save(f, y)
        meta = dict(name=tag,
                    target_type=ttype,
                    num_instances=int(X.shape[0]),
                    num_features=int(X.shape[1]),
                    num_unique_labels=int(np.unique(y).size),
                    cat_idx=np.where(np.asarray(cat_mask, bool))[0].tolist(),
                    feature_names=Xdf.columns.tolist())
        json.dump(meta, open(out / "metadata.json", "w"), indent=2)
        return tag, True, "done"
    except Exception as e:
        return tag, False, str(e)

def _args():
    p = argparse.ArgumentParser()
    p.add_argument("group", nargs="?", help="datasets.yaml group")
    p.add_argument("-c", "--config", default=str(CFG))
    p.add_argument("--one-tag")
    p.add_argument("-j", "--jobs", type=int, default=min(8, mp.cpu_count()))
    return p.parse_args()

def main():
    a = _args()
    cfg = yaml.safe_load(open(a.config))
    if a.one_tag:
        tag, ok, msg = _download(a.one_tag)
        print(f"[{'OK' if ok else 'FAIL'}] {tag}: {msg}")
        sys.exit(0 if ok else 1)
    if a.group not in cfg:
        raise KeyError(a.group)
    tags: List[str] = cfg[a.group]
    print(f"downloading {len(tags)} tasks → {DATA}")
    with mp.Pool(a.jobs) as pool:
        res = list(tqdm(pool.imap_unordered(_download, tags), total=len(tags)))
    ok = sum(o for _, o, _ in res)
    err = len(res) - ok
    print(f"\n✔ {ok} done   ✗ {err} failed\n")
    for t, o, m in sorted(res):
        if not o:
            print(f" {t} → {m}")
    sys.exit(0 if err == 0 else 1)

if __name__ == "__main__":
    main()
