# coding: utf-8
"""抓取 19 个 CSI/中证/SZ/海外指数 → data/ifind_cache/index_proxy/{code}.parquet.

数据源: 同花顺 iFinD 直连 API
  POST https://quantapi.51ifind.com/api/v1/date_sequence
- 19 codes × 9 indicators (Open/High/Low/Close/Settle/Vol/Amt + Name)
- 2018-01-01 ~ 2026-06-30 (最长可获取)
- 单次 POST 拿全部 19 codes 全期
- access_token 优先从 IFIND_ACCESS_TOKEN env 取，否则从 ~/.agents/skills/ifind/mcp_config.json

输出:
  data/ifind_cache/index_proxy/{code}.parquet
  ├── obs_date
  ├── name
  ├── pre_close, open, high, low, close
  ├── settle, vol, trans_amt
  ├── src = 'ifind.quantapi.date_sequence'

Usage:
  python3.11 scripts/fetch_proxy_indices.py
  python3.11 scripts/fetch_proxy_indices.py --codes 000688.SH,000906.SH  # 子集
  python3.11 scripts/fetch_proxy_indices.py --start 2018-01-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib3
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_CODES = [
    "000688.SH",   # 科创50
    "000906.SH",   # 中证800
    "399005.SZ",   # 深证红利
    "399324.SZ",   # 深证红利潜力
    "399998.SZ",   # 待确认（待edb search）
    "931450.CSI",  # 中证消费主题
    "931865.CSI",  # 中证半导体
    "399989.SZ",   # 中证医疗
    "399987.SZ",   # 中证酒
    "931151.CSI",  # 中证光伏产业
    "H30269.CSI",  # 中证智能制造
    "980017.SZ",   # 国证半导体芯片
    "HSTECH.HK",   # 恒生科技
    "NDX.GI",      # 纳斯达克100
    "931152.CSI",  # 中证创新药
    "930850.CSI",  # 中证智能制造（补）
    "000922.CSI",  # 中证红利
    "399810.SZ",   # 中证煤炭
    "399976.SZ",   # 中证新能源车
]

INDICATORS = [
    "ths_index_short_name_index",
    "ths_pre_close_index",
    "ths_open_price_index",
    "ths_high_price_index",
    "ths_low_index",
    "ths_close_price_index",
    "ths_settle_index",
    "ths_vol_index",
    "ths_trans_amt_index",
]

API_URL = "https://quantapi.51ifind.com/api/v1/date_sequence"
SKILL_CONFIG = Path.home() / ".agents/skills/ifind/mcp_config.json"
CACHE_DIR = PROJECT_ROOT / "data/ifind_cache/index_proxy"


def load_token() -> str:
    env = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
    if env:
        return env
    if SKILL_CONFIG.exists():
        cfg = json.loads(SKILL_CONFIG.read_text(encoding="utf-8"))
        tok = cfg.get("auth_token", "").strip()
        if tok:
            return tok
    raise RuntimeError(
        "access_token 未找到。请设置环境变量 IFIND_ACCESS_TOKEN 或确认 "
        f"{SKILL_CONFIG} 存在并含 auth_token"
    )


def fetch_one_window(
    token: str,
    codes: list[str],
    start: str,
    end: str,
    indicators: list[str],
    max_retries: int = 3,
    timeout: int = 120,
) -> dict:
    """单次 POST 全部 codes × 选定 indicators; 返回 API JSON。

    试用 token 受限:
      - 数据起点 ≥2021-07-01 (5 年窗口)
      - 单次 rows ≤ ~700 (依 指标数 × 月数 × 代码数)
      - 实测: 2 indicators × 11 月 × 19 codes = 4047 ✓ (api 名额记号: rowQuota)
    """
    payload = {
        "codes": ",".join(codes),
        "startdate": start,
        "enddate": end,
        "indipara": [
            {"indicator": ind, "indiparams": [""]}
            for ind in indicators
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "access_token": token,
    }
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                verify=False,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errorcode", 0) != 0:
                raise RuntimeError(
                    f"API errorcode={data.get('errorcode')} "
                    f"errmsg={data.get('errmsg')}"
                )
            return data
        except Exception as exc:
            last_err = exc
            logging.warning(
                "attempt %d/%d 失败: %s; 重试中", attempt, max_retries, exc
            )
            time.sleep(2 * attempt)
    raise RuntimeError(f"fetch 失败: {last_err}")


def parse_tables_to_frames(data: dict) -> dict[str, pd.DataFrame]:
    """把 API 返回的 tables 转成 {code: DataFrame}.

    API 返回结构:
      tables: [
        {
          'thscode': '000688.SH',
          'time':     ['2024-01-02', ...],
          'table':    {'indicator_name': [vals...], ...}
        },
        ...
      ]
    """
    out: dict[str, pd.DataFrame] = {}
    for entry in data.get("tables", []):
        code = entry.get("thscode")
        time_arr = entry.get("time", [])
        tbl = entry.get("table", {}) or {}
        if not code or not time_arr:
            continue
        df = pd.DataFrame({"obs_date": pd.to_datetime(time_arr)})
        for ind, vals in tbl.items():
            col = ind.replace("ths_", "").replace("_index", "").strip("_")
            df[col] = vals
        df = df.rename(
            columns={
                "index_short_name": "name",
                "pre_close": "pre_close",
                "open_price": "open",
                "high_price": "high",
                "low": "low",
                "close_price": "close",
                "settle": "settle",
                "vol": "vol",
                "trans_amt": "trans_amt",
            }
        )
        df["src"] = "ifind.quantapi.date_sequence"
        df["code"] = code
        df = df.sort_values("obs_date").reset_index(drop=True)
        out[code] = df
    return out


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--codes", default=",".join(DEFAULT_CODES),
        help="comma-separated iFinD codes，默认全 19"
    )
    parser.add_argument("--start", default="2021-07-01",
                        help="开始 YYYY-MM-DD (试用 token 受限 ≥2021-07-01)")
    parser.add_argument("--end", default="2026-06-30", help="结束 YYYY-MM-DD")
    parser.add_argument("--chunk-months", type=int, default=11,
                        help="按 N 月分块抓取 (默认 11，留安全边; 实测 ≤12 月尚可)")
    parser.add_argument("--inds", default="name,close",
                        help="指标子集 (逗号分隔). 选项: name,pre_close,open,high,low,close,settle,vol,trans_amt. "
                             "默认 'name,close'; 多指标会缩 chunk 体积上限.")
    parser.add_argument("--out-dir", default=str(CACHE_DIR),
                        help=f"输出目录，默认 {CACHE_DIR}")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    logging.info("开始抓取 %d 个 codes，起止 %s ~ %s", len(codes), args.start, args.end)

    token = load_token()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[tuple[str, str]]
    if args.chunk_months > 0:
        chunks = []
        cur = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        while cur < end:
            nxt = (cur + pd.DateOffset(months=args.chunk_months))
            if nxt > end:
                nxt = end
            chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            cur = nxt + pd.DateOffset(days=1)
        logging.info("分 %d 块抓取 (每块 ≤ %d 月)", len(chunks), args.chunk_months)
    else:
        chunks = [(args.start, args.end)]
        logging.info("单块抓取，依赖 API ≤12 月自动上限")

    merged: dict[str, list[pd.DataFrame]] = {c: [] for c in codes}

    # 解析用户指定的指标子集
    wanted_inds = [s.strip() for s in args.inds.split(",") if s.strip()]
    ind_name_map = {
        "name":       "ths_index_short_name_index",
        "pre_close":  "ths_pre_close_index",
        "open":       "ths_open_price_index",
        "high":       "ths_high_price_index",
        "low":        "ths_low_index",
        "close":      "ths_close_price_index",
        "settle":     "ths_settle_index",
        "vol":        "ths_vol_index",
        "trans_amt":  "ths_trans_amt_index",
    }
    selected_api_inds = [ind_name_map[k] for k in wanted_inds if k in ind_name_map]
    if not selected_api_inds:
        raise ValueError(f"--inds 无合法选项: {args.inds}")
    logging.info("请求指标: %s", selected_api_inds)

    for i, (cs, ce) in enumerate(chunks):
        if args.chunk_months:
            logging.info("块 %d/%d: %s ~ %s", i + 1, len(chunks), cs, ce)
        data = fetch_one_window(token, codes, cs, ce, selected_api_inds)
        frames = parse_tables_to_frames(data)
        for code in codes:
            if code in frames:
                merged[code].append(frames[code])
            else:
                logging.warning("块 %s 里 code %s 无返回", i + 1, code)

    # 合并 + 落盘
    summary_rows = []
    for code in codes:
        parts = merged[code]
        if not parts:
            logging.error("code %s 任何块都无返回", code)
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["obs_date"]).sort_values("obs_date")
        df = df.reset_index(drop=True)

        # 对每列尝试转数值
        for col in ("pre_close", "open", "high", "low", "close",
                    "settle", "vol", "trans_amt"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        cache_path = out_dir / f"{code.replace('.', '_')}.parquet"
        df.to_parquet(cache_path, index=False)
        logging.info(
            "✓ %s (%s) → %s (rows=%d, [%s .. %s], close NaN%%=%.2f%%)",
            code, df["name"].iloc[0] if "name" in df and len(df) else "?",
            cache_path.name, len(df),
            df["obs_date"].min().date(), df["obs_date"].max().date(),
            df["close"].isna().mean() * 100 if "close" in df else -1,
        )
        summary_rows.append({
            "code": code,
            "name": df["name"].iloc[0] if "name" in df and len(df) else "",
            "rows": len(df),
            "start": str(df["obs_date"].min().date()),
            "end":   str(df["obs_date"].max().date()),
            "close_nan_pct": round(df["close"].isna().mean() * 100, 2)
                              if "close" in df else None,
        })

    summary_path = out_dir / "_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logging.info("Summary → %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
