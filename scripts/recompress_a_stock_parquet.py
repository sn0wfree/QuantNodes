#!/usr/bin/env python3.11
# coding=utf-8
"""
recompress_a_stock_parquet.py - 用最高压缩等级重写 A 股 parquet 缓存.

polars write_parquet 默认 zstd level 3, 此脚本用 level 22 (zstd 上限)
重写以最大化压缩率 (解压速度影响极小, polars/arrow 解压时按需解).

用法::

    python3.11 scripts/recompress_a_stock_parquet.py
    python3.11 scripts/recompress_a_stock_parquet.py --path data/cache/full_a_2025_2026.parquet
    python3.11 scripts/recompress_a_stock_parquet.py --codec brotli --level 11
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data/cache/full_a_2025_2026.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用最高压缩等级重写 A 股 parquet 缓存",
    )
    parser.add_argument("--path", type=str, default=str(DEFAULT_PATH),
                        help="parquet 文件路径 (default: data/cache/full_a_2025_2026.parquet)")
    parser.add_argument("--codec", type=str, default="zstd",
                        choices=["zstd", "brotli", "gzip"],
                        help="压缩算法 (default: zstd)")
    parser.add_argument("--level", type=int, default=None,
                        help="压缩等级 (zstd:1-22, brotli:0-11, gzip:0-9); 留空则用 codec 最高档")
    return parser.parse_args()


CODEC_MAX = {"zstd": 22, "brotli": 11, "gzip": 9}


def main() -> None:
    args = parse_args()
    src = Path(args.path)
    if not src.exists():
        raise FileNotFoundError(f"parquet not found: {src}")

    level = args.level if args.level is not None else CODEC_MAX[args.codec]
    codec_max = CODEC_MAX[args.codec]
    if not (0 <= level <= codec_max):
        raise ValueError(f"--level {level} 超出 {args.codec} 范围 [0, {codec_max}]")

    size_before = src.stat().st_size

    import polars as pl

    print(f"{'='*60}")
    print("Parquet 重压缩")
    print(f"{'='*60}")
    print(f"  src    : {src}")
    print(f"  codec  : {args.codec} level={level} (max={codec_max})")

    df = pl.read_parquet(src)
    print(f"  rows   : {df.height:,}")
    print(f"  cols   : {df.columns}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="recomp_", dir=src.parent))
    tmp_out = tmp_dir / src.name
    try:
        df.write_parquet(tmp_out, compression=args.codec, compression_level=level)
        size_after = tmp_out.stat().st_size
        ratio = size_before / size_after
        saved_mb = (size_before - size_after) / 1024 / 1024

        shutil.move(str(tmp_out), str(src))
        print(f"  before : {size_before/1024/1024:.2f} MB")
        print(f"  after  : {size_after/1024/1024:.2f} MB")
        print(f"  ratio  : {ratio:.2f}x (节省 {saved_mb:.2f} MB)")
        print(f"{'='*60}")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
