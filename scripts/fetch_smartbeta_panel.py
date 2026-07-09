# coding=utf-8
"""拉取 12 只 Smart β ETF 净值, 落盘到 data/real/etf_nav_smartbeta_*.parquet.

数据源: Tencent web.ifzq.gtimg.cn (与 v3 共用)

12 只 Smart β:
  风格组 (5):
    510300 (HS300 大盘), 510500 (CSI500 中盘), 159915 (创业板), 588000 (科创50), 510880 (红利)
  Smart β 工具 (7):
    512890 (红利低波), 512260 (300 低波), 512040 (国泰价值), 515900 (中证质量),
    159786 (现金流), 515080 (中信红利), 515100 (红利低波100)

落盘结构:
    data/real/
    ├── etf_nav_smartbeta_2018-01-01_2026-06-30.parquet  # 12 列 × ~2058 行
    ├── per_etf_smartbeta/<code>.parquet                  # per-ETF 缓存
    └── smartbeta_fetch_log.json                          # 成功/失败日志
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation.common.data_eastmoney import (
    fetch_one_etf_eastmoney,
)


# 12 只 Smart β ETF
SMART_BETA_ETFS: list[str] = [
    # 风格组 (5)
    "510300",   # HS300 大盘
    "510500",   # CSI500 中盘
    "159915",   # 创业板
    "588000",   # 科创50
    "510880",   # 华泰柏瑞红利
    # Smart β 工具 (7)
    "512890",   # 红利低波
    "512260",   # 300 低波
    "515900",   # 中证质量
    "512040",   # 国泰价值
    "159786",   # 现金流
    "515080",   # 中信红利
    "515100",   # 红利低波 100
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取 12 只 Smart β ETF 真实数据 (Tencent 行情)")
    parser.add_argument("--start", default="2018-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-30", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "real"),
        help="落盘根目录",
    )
    parser.add_argument("--refresh", action="store_true", help="忽略缓存重拉")
    parser.add_argument("--sleep-ms", type=int, default=150, help="每请求间隔 ms")
    parser.add_argument("--codes", nargs="*", default=None, help="指定子集 codes")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("fetch_smartbeta_panel")

    out_dir = Path(args.out_dir)
    per_etf_dir = out_dir / "per_etf_smartbeta"
    per_etf_dir.mkdir(parents=True, exist_ok=True)

    codes = args.codes or SMART_BETA_ETFS
    n = len(codes)
    log.info("准备拉取 %d 只 Smart β ETF, 范围 %s ~ %s", n, args.start, args.end)
    log.info("落盘目录: %s", out_dir)

    fetched: dict[str, int] = {}
    failed: list[str] = []
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        cache_path = per_etf_dir / f"{code}.parquet"
        if cache_path.exists() and not args.refresh:
            try:
                cached = pd.read_parquet(cache_path)
                if len(cached) > 0:
                    fetched[code] = len(cached)
                    log.info("[%d/%d] %s 缓存命中 (%d 行)", i, n, code, len(cached))
                    continue
            except Exception as e:
                log.warning("[%d/%d] %s 缓存损坏, 重拉: %s", i, n, code, e)

        try:
            series = fetch_one_etf_eastmoney(
                code, args.start, args.end, args.sleep_ms,
            )
            if series.empty:
                log.warning("[%d/%d] %s 数据为空 (可能晚于 %s 上市)", i, n, code, args.start)
                failed.append(code)
                continue
            series.to_frame().to_parquet(cache_path)
            fetched[code] = len(series)
            log.info("[%d/%d] %s 成功 (%d 行) %.3f -> %.3f",
                     i, n, code, len(series), series.iloc[0], series.iloc[-1])
        except Exception as e:
            log.error("[%d/%d] %s 失败: %s", i, n, code, e)
            failed.append(code)

    # 合并为面板
    if fetched:
        log.info("合并 %d 个 parquet 为面板...", len(fetched))
        dfs = []
        for code in fetched:
            s = pd.read_parquet(per_etf_dir / f"{code}.parquet")
            dfs.append(s)
        panel = pd.concat(dfs, axis=1).sort_index()
        # 限日期范围
        start_dt = pd.Timestamp(args.start)
        end_dt = pd.Timestamp(args.end)
        panel = panel.loc[(panel.index >= start_dt) & (panel.index <= end_dt)]
        # ffill 限 5
        panel = panel.ffill(limit=5)
        # 写出主面板
        panel_path = out_dir / f"etf_nav_smartbeta_{args.start}_{args.end}.parquet"
        panel.to_parquet(panel_path)
        log.info("面板落盘: %s, shape=%s", panel_path, panel.shape)

    # 写 fetch log
    log_data = {
        "start": args.start,
        "end": args.end,
        "fetched": fetched,
        "failed": failed,
        "total": n,
        "success": len(fetched),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    log_path = out_dir / "smartbeta_fetch_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    log.info("日志落盘: %s", log_path)

    log.info("=" * 60)
    log.info("完成: 成功 %d / 失败 %d / 总数 %d, 耗时 %.1fs",
             len(fetched), len(failed), n, time.time() - t0)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
