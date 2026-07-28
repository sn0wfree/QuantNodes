"""Build KNN graph and save adjacency + diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from ca_gcp import build_knn_graph  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--method", choices=["correlation", "sector", "random"], default="correlation")
    parser.add_argument("--train-end", default="2021-04-12")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1)
    df = df.ffill().fillna(0.0)

    train = df.loc[: args.train_end]
    print(f"Train: {train.shape}, date range: {train.index.min()} -> {train.index.max()}")

    A_norm, neighbors, codes = build_knn_graph(train, k=args.k, method=args.method)

    np.save(OUT_DIR / "A_norm.npy", A_norm)
    pd.DataFrame(A_norm, index=codes, columns=codes).to_parquet(OUT_DIR / "A_norm.parquet")

    nbr_rows = []
    for i, code in enumerate(codes):
        for j in neighbors[i]:
            nbr_rows.append({"target": code, "source": codes[j], "neighbor_idx": j})
    pd.DataFrame(nbr_rows).to_csv(OUT_DIR / "neighbors.csv", index=False)

    print(f"Saved A_norm ({A_norm.shape}) and {len(nbr_rows)} neighbor pairs to {OUT_DIR}")


if __name__ == "__main__":
    main()