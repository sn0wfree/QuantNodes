#!/usr/bin/env python3.11
"""从 iFinD MCP 获取缺失的 ETF OHLCV 数据.

使用 iFinD MCP 接口查询 C 类和 D 类 ETF 的缺失数据，
然后合并到现有的 OHLCV 数据集。

使用方法:
    python3.11 scripts/fetch_ifind_data.py
"""
from __future__ import annotations

import sys
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 添加 iFinD MCP 路径
IFIND_PATH = Path("/home/ll/Public/ifind-finance-data-1.1.0")
sys.path.insert(0, str(IFIND_PATH))

# call.py 需要从其所在目录读取 mcp_config.json
import os
_orig_cwd = os.getcwd()
os.chdir(str(IFIND_PATH))
from call import call
os.chdir(_orig_cwd)

REPO = Path(__file__).resolve().parents[1]
OHLCV_PATH = REPO / "data" / "real" / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
OHLCV_BACKUP_PATH = REPO / "data" / "real" / "etf_ohlcv_2018-01-01_2026-06-30_adjusted_backup.parquet"


# C 类 ETF: NaN 2-8%, 需要补充
C_CLASS_ETFS = [
    ('512980', '2017-05-23', '2018-01-18'),
    ('512200', '2017-05-23', '2017-09-22'),
    ('512400', '2017-05-23', '2017-08-31'),
    ('511260', '2017-05-23', '2018-08-17'),
    ('512800', '2017-05-23', '2017-08-02'),
]

# D 类 ETF: NaN 20-46%, 可选补充
D_CLASS_ETFS = [
    ('159766', '2017-05-23', '2026-07-08'),
    ('159740', '2017-05-23', '2026-07-08'),
    ('513010', '2017-05-23', '2026-07-08'),
    ('515790', '2017-05-23', '2026-07-08'),
    ('588000', '2017-05-23', '2026-07-08'),
    ('513300', '2017-05-23', '2026-07-08'),
    ('159996', '2017-05-23', '2026-07-08'),
    ('515030', '2017-05-23', '2026-07-08'),
    ('515220', '2017-05-23', '2026-07-08'),
    ('159981', '2017-05-23', '2026-07-08'),
    ('159985', '2017-05-23', '2026-07-08'),
    ('515050', '2017-05-23', '2026-07-08'),
    ('515880', '2017-05-23', '2026-07-08'),
    ('513520', '2017-05-23', '2026-07-08'),
    ('513880', '2017-05-23', '2026-07-08'),
    ('512170', '2017-05-23', '2026-07-08'),
    ('512480', '2017-05-23', '2026-07-08'),
    ('512760', '2017-05-23', '2026-07-08'),
    ('512690', '2017-05-23', '2026-07-08'),
]


def parse_ifind_markdown(text: str) -> pd.DataFrame | None:
    """解析 iFinD 返回的 markdown 表格数据.

    格式:
    |证券代码|证券简称|日期|开盘价（单位：元）|最低价（单位：元）|收盘价（单位：元）|最高价（单位：元）|成交量|
    |---|---|---|---|---|---|---|---|
    |512980.SH|传媒ETF广发|20200115|0.924|0.911|0.922|0.93|4520.22万|
    """
    if not text:
        return None

    # 找到表格行
    lines = text.strip().split('\n')
    table_lines = [l for l in lines if l.startswith('|') and '---' not in l]

    if len(table_lines) < 2:
        return None

    # 解析表头
    header = [h.strip() for h in table_lines[0].split('|') if h.strip()]

    # 解析数据行
    rows = []
    for line in table_lines[1:]:
        cells = [c.strip() for c in line.split('|') if c.strip()]
        if len(cells) >= 8:
            rows.append(cells)

    if not rows:
        return None

    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=header)

    # 转换日期
    if '日期' in df.columns:
        df['date'] = pd.to_datetime(df['日期'], format='%Y%m%d')
        df = df.set_index('date')

    # 转换数值
    for col in ['开盘价（单位：元）', '最低价（单位：元）', '收盘价（单位：元）', '最高价（单位：元）']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 转换成交量
    if '成交量' in df.columns:
        def parse_volume(v):
            if isinstance(v, (int, float)):
                return float(v)
            v = str(v).strip()
            if '亿' in v:
                return float(v.replace('亿', '')) * 1e8
            elif '万' in v:
                return float(v.replace('万', '')) * 1e4
            else:
                try:
                    return float(v)
                except:
                    return np.nan
        df['volume'] = df['成交量'].apply(parse_volume)

    # 重命名列
    rename_map = {
        '开盘价（单位：元）': 'open',
        '最低价（单位：元）': 'low',
        '收盘价（单位：元）': 'close',
        '最高价（单位：元）': 'high',
    }
    df = df.rename(columns=rename_map)

    # 选择需要的列
    cols = ['open', 'high', 'low', 'close', 'volume']
    available = [c for c in cols if c in df.columns]
    return df[available]


def fetch_etf_data(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """查询单个 ETF 的 OHLCV 数据."""
    # 添加 .SH 后缀 (上海交易所)
    if code.startswith('5') or code.startswith('6'):
        ifind_code = f"{code}.SH"
    elif code.startswith('1'):
        ifind_code = f"{code}.SZ"
    else:
        ifind_code = f"{code}.SH"

    query = f"{ifind_code} {start_date}至{end_date}的每日开盘价、最高价、最低价、收盘价、成交量"

    try:
        result = call('stock', 'get_stock_info', {'query': query})
        if not result['ok']:
            print(f"    ❌ API 调用失败: {result.get('error', '未知错误')}")
            return None

        # 解析返回数据
        data = result['data']
        if 'result' not in data or 'content' not in data['result']:
            print(f"    ❌ 返回格式错误")
            return None

        text = data['result']['content'][0].get('text', '')
        # 解析 JSON 字符串
        try:
            inner = json.loads(text)
            answer = inner.get('data', {}).get('answer', '')
        except:
            answer = text

        # 检查是否为空
        if '工具调用结果为空' in answer:
            print(f"    ⚠️ 无数据 (ETF 可能未上市)")
            return None

        # 解析 markdown 表格
        df = parse_ifind_markdown(answer)
        if df is None or df.empty:
            print(f"    ⚠️ 解析失败或无数据")
            return None

        return df

    except Exception as e:
        print(f"    ❌ 异常: {e}")
        return None


def merge_ifind_data(
    ohlcv: pd.DataFrame,
    code: str,
    ifind_df: pd.DataFrame,
) -> pd.DataFrame:
    """将 iFinD 数据合并到 OHLCV 数据.

    只填充 NaN 位置，不覆盖现有数据.
    """
    if ifind_df is None or ifind_df.empty:
        return ohlcv

    # 获取原始数据
    original = ohlcv[code].copy()

    # 对齐日期
    common_idx = original.index.intersection(ifind_df.index)
    if len(common_idx) == 0:
        return ohlcv

    # 逐列填充 NaN
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in ifind_df.columns and col in original.columns:
            nan_mask = original[col].isna()
            fill_idx = nan_mask.index[nan_mask].intersection(common_idx)
            if len(fill_idx) > 0:
                original.loc[fill_idx, col] = ifind_df.loc[fill_idx, col]

    ohlcv[code] = original
    return ohlcv


def fetch_and_merge_etfs(
    etf_list: list[tuple[str, str, str]],
    ohlcv: pd.DataFrame,
    category: str,
) -> tuple[pd.DataFrame, dict]:
    """查询并合并 ETF 数据."""
    results = {}
    for code, start_date, end_date in etf_list:
        print(f"\n  查询 {code} ({start_date} ~ {end_date})...")

        # 查询数据
        ifind_df = fetch_etf_data(code, start_date, end_date)

        if ifind_df is not None and not ifind_df.empty:
            # 合并数据
            ohlcv = merge_ifind_data(ohlcv, code, ifind_df)
            results[code] = {
                'status': 'success',
                'rows': len(ifind_df),
                'date_range': f"{ifind_df.index[0]} ~ {ifind_df.index[-1]}",
            }
            print(f"    ✅ 成功: {len(ifind_df)} 行")
        else:
            results[code] = {'status': 'no_data'}
            print(f"    ⚠️ 无数据")

        # 控制请求频率 (免费用户每秒最多2个请求)
        time.sleep(0.5)

    return ohlcv, results


def main():
    """主函数."""
    print("=" * 60)
    print("从 iFinD MCP 获取缺失的 ETF 数据")
    print("=" * 60)

    # 备份原始数据
    if not OHLCV_BACKUP_PATH.exists():
        print("\n备份原始数据...")
        import shutil
        shutil.copy2(OHLCV_PATH, OHLCV_BACKUP_PATH)
        print(f"  备份到: {OHLCV_BACKUP_PATH}")

    # 加载 OHLCV 数据
    print("\n加载 OHLCV 数据...")
    ohlcv = pd.read_parquet(OHLCV_PATH)
    print(f"  Shape: {ohlcv.shape}")

    # 查询 C 类 ETF
    print("\n" + "=" * 60)
    print("Phase 1: 查询 C 类 ETF (5 个)")
    print("=" * 60)
    ohlcv, c_results = fetch_and_merge_etfs(C_CLASS_ETFS, ohlcv, 'C')

    # 查询 D 类 ETF
    print("\n" + "=" * 60)
    print("Phase 2: 查询 D 类 ETF (19 个)")
    print("=" * 60)
    ohlcv, d_results = fetch_and_merge_etfs(D_CLASS_ETFS, ohlcv, 'D')

    # 保存更新后的数据
    print("\n" + "=" * 60)
    print("保存更新后的数据")
    print("=" * 60)
    ohlcv.to_parquet(OHLCV_PATH)
    print(f"  保存到: {OHLCV_PATH}")

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    print("\nC 类 ETF:")
    for code, result in c_results.items():
        status = result['status']
        if status == 'success':
            print(f"  {code}: ✅ {result['rows']} 行 ({result['date_range']})")
        else:
            print(f"  {code}: ⚠️ 无数据")

    print("\nD 类 ETF:")
    d_success = sum(1 for r in d_results.values() if r['status'] == 'success')
    d_no_data = sum(1 for r in d_results.values() if r['status'] == 'no_data')
    print(f"  成功: {d_success} 个")
    print(f"  无数据: {d_no_data} 个")

    # 检查更新后的 NaN 比例
    print("\n" + "=" * 60)
    print("更新后的数据质量")
    print("=" * 60)

    # 检查所有 ETF 的 NaN 比例
    codes = sorted(ohlcv.columns.get_level_values(0).unique())
    nan_ratios = []
    for code in codes:
        nan_ratio = ohlcv[code].isna().mean().mean()
        nan_ratios.append((code, nan_ratio))

    # 统计
    valid_count = sum(1 for _, r in nan_ratios if r < 0.20)
    invalid_count = sum(1 for _, r in nan_ratios if r >= 0.20)
    print(f"  有效 ETF (NaN < 20%): {valid_count} 个")
    print(f"  无效 ETF (NaN >= 20%): {invalid_count} 个")

    # 显示 NaN > 10% 的 ETF
    print("\n  NaN > 10% 的 ETF:")
    for code, ratio in sorted(nan_ratios, key=lambda x: x[1], reverse=True):
        if ratio > 0.10:
            print(f"    {code}: {ratio:.2%}")

    print("\n完成!")


if __name__ == "__main__":
    main()
