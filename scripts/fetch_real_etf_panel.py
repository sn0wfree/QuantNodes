# coding=utf-8
"""拉取 43 支 ETF 净值 (2018-2025) 落盘到 data/real/etf_nav_2018_2025.parquet.

数据源: Tencent web.ifzq.gtimg.cn (无需 akshare).
速率: 7 req/s 实测, 加 150ms sleep 留余量.
43 支 × 5 块 ≈ 215 reqs, 约 30s 一次拉完.

落盘结构:
    data/real/
    ├── etf_nav_2018_2025.parquet     # 主面板 (43 列 × ~1700 行)
    ├── per_etf/<code>.parquet        # per-ETF 缓存 (失败可重拉单支)
    └── fetch_log.json                # 哪支成功/失败/缺失
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# 让脚本可以独立运行 (不依赖包安装)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation import DEFAULT_POOL
from QuantNodes.strategy.momentum_etf_rotation.common.data_tencent import (
    fetch_one_etf_tencent,
    write_fetch_log,
)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取 43 支 ETF 真实数据 (Tencent 行情)")
    parser.add_argument("--start", default="2018-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2025-07-06", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "real"),
        help="落盘根目录",
    )
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重拉")
    parser.add_argument("--sleep-ms", type=int, default=150, help="每请求间隔 ms")
    parser.add_argument("--codes", nargs="*", default=None,
                         help="指定子集 codes, 缺省=DEFAULT_POOL 全部 43 支")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("fetch_real_etf_panel")

    out_dir = Path(args.out_dir)
    per_etf_dir = out_dir / "per_etf"
    per_etf_dir.mkdir(parents=True, exist_ok=True)

    codes = args.codes or list(DEFAULT_POOL.codes)
    # 默认追加 511260 (国泰上证 10 年期国债 ETF) 作为 80/20 固收+ 的债券部分
    if not args.codes and "511260" not in codes:
        codes = list(codes) + ["511260"]
        log.info("自动追加 511260 (10 年国债 ETF)")
    n = len(codes)
    log.info("准备拉取 %d 支 ETF, 范围 %s ~ %s", n, args.start, args.end)
    log.info("落盘目录: %s", out_dir)

    fetched: dict[str, int] = {}
    failed: list[str] = []
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        per_path = per_etf_dir / f"{code}.parquet"
        if not args.refresh and per_path.exists():
            try:
                df = pd.read_parquet(per_path)
                if "close" in df.columns and not df.empty:
                    fetched[code] = len(df)
                    log.info("[%d/%d] %s 走缓存 (%d 行)", i, n, code, len(df))
                    continue
            except Exception as exc:
                log.warning("缓存读取失败 %s: %s, 重拉", code, exc)
        try:
            s = fetch_one_etf_tencent(code, args.start, args.end, sleep_ms=args.sleep_ms)
            if not s.empty:
                pd.DataFrame({"close": s}).to_parquet(per_path)
                fetched[code] = len(s)
                log.info("[%d/%d] %s 拉到 %d 行 (%s ~ %s)",
                          i, n, code, len(s), s.index[0].date(), s.index[-1].date())
            else:
                failed.append(code)
                log.info("[%d/%d] %s 无数据 (可能 2018 后上市)", i, n, code)
        except Exception as exc:
            failed.append(code)
            log.warning("[%d/%d] %s 失败: %s", i, n, code, exc)

    elapsed = time.time() - t0

    # 主面板: 拼接所有 per-ETF
    if not fetched:
        log.error("无任何成功数据, 不写主面板")
        return 1

    series_map: dict[str, pd.Series] = {}
    for code in fetched:
        try:
            df = pd.read_parquet(per_etf_dir / f"{code}.parquet")
            series_map[code] = df["close"].rename(code)
        except Exception as exc:
            log.warning("读 %s 失败: %s", code, exc)
            failed.append(code)

    panel = pd.concat(series_map.values(), axis=1).sort_index()
    panel = panel.ffill(limit=5)
    # 仅在 fetch 包含 DEFAULT_POOL (全量) 时写主面板
    panel_path = out_dir / f"etf_nav_{args.start}_{args.end}.parquet"
    is_full_fetch = not args.codes
    if is_full_fetch:
        panel.to_parquet(panel_path)
        log.info("主面板落盘: %s (shape=%s)", panel_path, panel.shape)
    else:
        log.info("--codes 模式: 不覆盖主面板 (避免单支重拉时丢失其他列)")

    # 写 fetch log
    log_path = out_dir / "fetch_log.json"
    write_fetch_log(fetched, failed, log_path)
    log.info("Fetch log: %s", log_path)

    log.info("=" * 60)
    log.info("完成: %d/%d 成功, %d 失败/缺失, 耗时 %.1fs",
              len(fetched), n, len(failed), elapsed)
    if failed:
        log.warning("失败/缺失: %s", ", ".join(failed))
    return 0 if not failed else 0  # 部分失败也返回 0 (主面板仍可读)


if __name__ == "__main__":
    raise SystemExit(main())
