# coding: utf-8
"""合并 19 个 iFinD proxy 指数 → 对齐 v56 面板索引 → 输出 pre/post NaN 表.

输入:
  data/ifind_cache/index_proxy/{code}.parquet
  data/high_freq_macro/v56_expanded_daily.parquet
  QuantNodes/strategy/momentum_etf_rotation/common/universe.py (ETFMeta: 44 ETF)

输出:
  data/high_freq_macro/v56_proxy_indices_daily.parquet
      合并 19 个 proxy 指数, 列名 (例: '000688.SH_科创50')
      索引: 与 v56_expanded_daily 对齐 (trading days)
      字段: close (主), 其他辅助
  data/high_freq_macro/_proxy_nan_table.csv
      pre/post 改造的 ETF NaN 状况对比表
  data/high_freq_macro/_proxy_etf_map.csv
      ETF → (真实对标 + proxy code + 是否被 iFinD 抓取)

用法:
  python3.11 scripts/build_proxy_panel.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_DIR = PROJECT_ROOT / "data/ifind_cache/index_proxy"
V56_PATH  = PROJECT_ROOT / "data/high_freq_macro/v56_expanded_daily.parquet"
OUT_PATH  = PROJECT_ROOT / "data/high_freq_macro/v56_proxy_indices_daily.parquet"
NAN_TABLE = PROJECT_ROOT / "data/high_freq_macro/_proxy_nan_table.csv"
ETF_MAP   = PROJECT_ROOT / "data/high_freq_macro/_proxy_etf_map.csv"

# ETF → 真实对标指数 + 用户提供的 iFinD 代码
# (code 列不在 v9 universe 中的 ETF 也会显示出来)
ETF_PROXY_MAP: list[dict] = [
    # 26 late-listing ETFs (按 pre-IPO 月数降序)
    {"code": "159786", "name": "深证红利ETF",  "ipo": "2021-08-10",
     "target": "深证红利",         "ifind_code": "399324.SZ", "src_proxy": None},
    {"code": "159766", "name": "黄金ETF基金", "ipo": "2021-07-27",
     "target": "黄金 (AU9999/SGE)", "ifind_code": None, "src_proxy": "v9:沪金指数"},
    {"code": "513010", "name": "港股通科技", "ipo": "2021-05-27",
     "target": "港股通科技 (已删)", "ifind_code": None, "src_proxy": None},
    {"code": "159740", "name": "恒生科技",  "ipo": "2021-05-31",
     "target": "恒生科技",         "ifind_code": "HSTECH.HK", "src_proxy": None},
    {"code": "515790", "name": "光伏ETF",   "ipo": "2020-12-22",
     "target": "中证光伏产业",     "ifind_code": "931151.CSI", "src_proxy": None},
    {"code": "588000", "name": "科创50ETF", "ipo": "2020-11-18",
     "target": "上证科创板50",     "ifind_code": "000688.SH", "src_proxy": None},
    {"code": "513300", "name": "纳斯达克100", "ipo": "2020-11-09",
     "target": "纳斯达克100",       "ifind_code": "NDX.GI", "src_proxy": None},
    {"code": "515100", "name": "智能制造ETF", "ipo": "2020-07-07",
     "target": "中证智能制造主题", "ifind_code": "930850.CSI", "src_proxy": None},
    {"code": "515220", "name": "煤炭ETF",   "ipo": "2020-03-04",
     "target": "中证煤炭",         "ifind_code": "399998.SZ", "src_proxy": None},
    {"code": "515030", "name": "新能车ETF", "ipo": "2020-03-06",
     "target": "中证新能源汽车",    "ifind_code": "399976.SZ", "src_proxy": None},
    {"code": "159996", "name": "主要消费ETF", "ipo": "2020-03-18",
     "target": "中证主要消费",      "ifind_code": "931450.CSI", "src_proxy": None},
    {"code": "159981", "name": "能源化工ETF", "ipo": "2020-01-21",
     "target": "能源化工期货主题", "ifind_code": None, "src_proxy": "v9:南华工业品指数"},
    {"code": "515080", "name": "创新药ETF", "ipo": "2019-12-31",
     "target": "中证创新药",       "ifind_code": "931152.CSI", "src_proxy": None},
    {"code": "159985", "name": "豆粕ETF",   "ipo": "2019-12-09",
     "target": "豆粕期货指数",      "ifind_code": None, "src_proxy": "v9:南华农产品指数"},
    {"code": "515900", "name": "中证800ETF", "ipo": "2019-12-20",
     "target": "中证800",          "ifind_code": "000906.SH", "src_proxy": None},
    {"code": "515050", "name": "5G通信ETF", "ipo": "2019-10-18",
     "target": "中证5G通信主题",    "ifind_code": None, "src_proxy": "v9:中证500指数"},
    {"code": "515880", "name": "通信设备ETF", "ipo": "2019-09-10",
     "target": "中证通信设备主题",  "ifind_code": None, "src_proxy": "v9:中证500指数"},
    {"code": "513880", "name": "日经225ETF", "ipo": "2019-06-27",
     "target": "日经225",          "ifind_code": None, "src_proxy": None},
    {"code": "513520", "name": "日经225ETF", "ipo": "2019-06-27",
     "target": "日经225",          "ifind_code": None, "src_proxy": None},
    {"code": "512170", "name": "医疗ETF",   "ipo": "2019-06-19",
     "target": "中证医疗",         "ifind_code": "399989.SZ", "src_proxy": None},
    {"code": "512760", "name": "半导体ETF", "ipo": "2019-06-14",
     "target": "国证半导体芯片",    "ifind_code": "980017.SZ", "src_proxy": "931865.CSI"},
    {"code": "512480", "name": "半导体ETF", "ipo": "2019-06-14",
     "target": "国证半导体芯片",    "ifind_code": "980017.SZ", "src_proxy": "931865.CSI"},
    {"code": "512690", "name": "酒ETF",     "ipo": "2019-05-08",
     "target": "中证酒",           "ifind_code": "399987.SZ", "src_proxy": None},
    {"code": "512890", "name": "红利低波ETF", "ipo": "2019-01-22",
     "target": "中证红利低波动",    "ifind_code": "H30269.CSI", "src_proxy": None},
    {"code": "512260", "name": "红利ETF",   "ipo": "2019-01-15",
     "target": "中证红利",         "ifind_code": "000922.CSI", "src_proxy": None},
    {"code": "512040", "name": "ESG ETF",  "ipo": "2018-12-03",
     "target": "沪深300 ESG",      "ifind_code": None, "src_proxy": "v9:沪深300"},
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_proxies() -> dict[str, pd.DataFrame]:
    """key = iFinD code, value = parquet DataFrame."""
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(CACHE_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        # file_name: 000688_SH.parquet → code: 000688.SH
        code = p.stem.replace("_", ".")
        df["code"] = code
        out[code] = df
    return out


def build_proxy_panel(proxies: dict[str, pd.DataFrame], base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """对齐到 base_index, 列名 = '{code}_{name}' 拼接."""
    cols = {}
    for code, df in proxies.items():
        name = df["name"].iloc[0] if "name" in df and len(df) else code
        short = f"{code}_{name}"
        # series indexed by obs_date
        s = pd.Series(
            pd.to_numeric(df["close"], errors="coerce").values,
            index=pd.DatetimeIndex(df["obs_date"]),
            name=short,
        )
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # reindex to base (NaN 留给前段)
        s = s.reindex(base_index)
        cols[short] = s
    return pd.DataFrame(cols, index=base_index)


def nan_status_table(
    etf_panel: pd.DataFrame, proxy_panel: pd.DataFrame,
) -> pd.DataFrame:
    """对每个 ETF 输出: 上市前/后段 NaN%, proxy code, target."""
    rows = []
    for entry in ETF_PROXY_MAP:
        code = entry["code"]
        ipo  = pd.Timestamp(entry["ipo"])
        if code not in etf_panel.columns:
            continue
        s = etf_panel[code]
        pre_nan  = s.loc[s.index < ipo].isna().mean() * 100
        post_nan = s.loc[s.index >= ipo].isna().mean() * 100
        # 如果该 ETF 有 iFinD proxy 可覆盖 post 段
        proxy = entry.get("ifind_code")
        post_target = entry.get("target")
        rows.append({
            "etf": code,
            "name": entry["name"],
            "ipo": ipo.strftime("%Y-%m-%d"),
            "target": post_target,
            "ifind_code": proxy if proxy else "—",
            "pre_ipo_nan_pct": round(pre_nan, 2),
            "post_ipo_nan_pct": round(post_nan, 2),
            "src_proxy": entry.get("src_proxy") or "",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    logging.info("加载 v56 面板作为基准索引...")
    v56 = pd.read_parquet(V56_PATH)
    v56.index = pd.DatetimeIndex(v56.index)
    base_index = v56.index
    logging.info("基准 trading days: %d, [%s .. %s]",
                len(base_index), base_index[0].date(), base_index[-1].date())

    logging.info("加载 19 个 iFinD proxy 指数 parquets...")
    proxies = load_proxies()
    for code, df in proxies.items():
        logging.info("  %s (%s): rows=%d", code, df["name"].iloc[0] if "name" in df else "?", len(df))

    logging.info("构建合并 proxy panel, 对齐到 v56 索引 (无前向填充)...")
    proxy_panel = build_proxy_panel(proxies, base_index)
    proxy_panel.to_parquet(OUT_PATH)
    logging.info("→ %s, shape=%s", OUT_PATH, proxy_panel.shape)

    # 输出 NaN 现状对比表
    logging.info("输出每只 ETF 的 NaN 状况...")
    nan_df = nan_status_table(v56, proxy_panel)
    nan_df.to_csv(NAN_TABLE, index=False)

    # 输出 ETF→proxy map
    etf_map_df = pd.DataFrame(ETF_PROXY_MAP)
    etf_map_df.to_csv(ETF_MAP, index=False)

    # 总结对比
    pre_avg  = nan_df["pre_ipo_nan_pct"].mean()
    post_avg = nan_df["post_ipo_nan_pct"].mean()
    full_nan_overall = v56.isna().mean().mean() * 100
    proxy_nan_overall = proxy_panel.isna().mean().mean() * 100
    logging.info("=== 总结 ===")
    logging.info("  v56 全期 daily NaN: %.2f%%", full_nan_overall)
    logging.info("  19 proxy 全期 daily NaN: %.2f%%", proxy_nan_overall)
    logging.info("  26 late-listing 平均 pre-IPO NaN: %.2f%%", pre_avg)
    logging.info("  26 late-listing 平均 post-IPO NaN: %.2f%%", post_avg)
    logging.info("  proxy 与晚上市 ETF 匹配个数: %d / %d",
                nan_df["ifind_code"].ne("—").sum(), len(nan_df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
