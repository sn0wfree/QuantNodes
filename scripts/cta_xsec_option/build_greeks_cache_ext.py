#!/usr/bin/env python3
# coding=utf-8
"""全量期权 Greeks cache 扩容 — 27品种池 × 2018-2026 (xsec_option IV 增强).

复用 scripts/build_option_greeks_cache.py 的反解内核 (_compute_chunk),
按品种分段落盘 (data/cache/option_greeks_ext/{product}.parquet):
    - 中断可续跑: 已有品种文件自动跳过
    - 全部完成后合并为 option_greeks_ext.parquet

用法:
    python3 scripts/cta_xsec_option/build_greeks_cache_ext.py            # 全量
    python3 scripts/cta_xsec_option/build_greeks_cache_ext.py M,I        # 指定品种
"""
from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec_cfg = importlib.util.spec_from_file_location(
    "xsec_config", Path(__file__).resolve().parent / "config.py"
)
_cfg = importlib.util.module_from_spec(_spec_cfg)
_spec_cfg.loader.exec_module(_cfg)

UNIVERSE = _cfg.UNIVERSE

_spec = importlib.util.spec_from_file_location("bogc", ROOT / "build_option_greeks_cache.py")
_bogc = importlib.util.module_from_spec(_spec)
sys.modules["bogc"] = _bogc
_spec.loader.exec_module(_bogc)

_compute_chunk = _bogc._compute_chunk

START = "2018-01-01"
END = "2026-07-21"
OUT_DIR = Path("/home/ll/Public/QuantNodes/data/cache/option_greeks_ext")
MERGED = OUT_DIR / "option_greeks_ext.parquet"
CHUNK_SYMS = _bogc.CHUNK_SYMS
COLUMNS = _bogc.COLUMNS


def _product_done(p: str) -> bool:
    f = OUT_DIR / f"{p}.parquet"
    return f.exists() and f.stat().st_size > 0


def _all_tasks(products: list[str]):
    loader = _bogc.CtaDataLoader() if hasattr(_bogc, "CtaDataLoader") else None
    if loader is None:
        from QuantNodes.strategy.cta.data_loader import CtaDataLoader
        loader = CtaDataLoader()
    tasks = {}
    for product in products:
        if _product_done(product):
            continue
        df = loader.load_v_all_options(products=[product], start=START, end=END)
        if df is None or df.empty:
            print(f"  [{product}] 无期权数据, 跳过")
            continue
        syms = sorted(df["symbol"].str.upper().unique().tolist())
        chunks = [syms[i:i + CHUNK_SYMS] for i in range(0, len(syms), CHUNK_SYMS)]
        tasks[product] = [(product, ch, START, END) for ch in chunks]
    return tasks


def main():
    products = sys.argv[1].split(",") if len(sys.argv) > 1 else list(UNIVERSE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks_by_product = _all_tasks(products)
    todo = {p: t for p, t in tasks_by_product.items() if t}
    n_total = sum(len(t) for t in todo.values())
    n_cores = max(1, min(n_total, 16))
    print(f"品种 {len(products)}, 待跑 {len(todo)} 个 / {n_total} chunks, cores={n_cores}")

    for product, prod_tasks in todo.items():
        t_prod = time.time()
        frames = []
        with ProcessPoolExecutor(max_workers=n_cores) as ex:
            futs = {ex.submit(_compute_chunk, t): t for t in prod_tasks}
            for fut in as_completed(futs):
                try:
                    frames.append(fut.result())
                except Exception as e:
                    print(f"  [{product}] chunk FAILED: {e}")
        if not frames:
            print(f"  [{product}] 无有效数据")
            continue
        out = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / f"{product}.parquet"
        out.to_parquet(out_path, index=False)
        valid_n = int(out["valid"].sum())
        print(
            f"  [{product}] rows={len(out)}, valid={valid_n} "
            f"({valid_n / len(out) * 100:.1f}%), "
            f"{time.time() - t_prod:.0f}s -> {out_path.name}",
            flush=True,
        )

    parts = [OUT_DIR / f"{p}.parquet" for p in UNIVERSE if _product_done(p)]
    if parts:
        merged = pd.concat(
            [pd.read_parquet(f) for f in parts], ignore_index=True
        )
        merged.to_parquet(MERGED, index=False)
        print(
            f"\n合并: {len(parts)} 品种, rows={len(merged)}, "
            f"valid={int(merged['valid'].sum())}, {time.time() - t0:.0f}s "
            f"-> {MERGED} ({MERGED.stat().st_size / 1e6:.0f} MB)"
        )


if __name__ == "__main__":
    main()
