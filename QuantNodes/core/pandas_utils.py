# coding=utf-8
"""
Pandas 工具函数

提供常用的 Pandas 数据处理工具函数，包括：
- Panel/DataFrame 转换
- 数据对齐
- 缺失值处理
- 数据重塑
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Any


def panel_to_dataframe(
    panel: Union[dict, List[pd.DataFrame]],
    orientation: str = "ftime"
) -> pd.DataFrame:
    """
    将 Panel（字典或 DataFrame 列表）转换为长格式 DataFrame

    Args:
        panel: 字典 {item: DataFrame} 或 DataFrame 列表
        orientation: "ftime"（时点在行） 或 "fitem"（item 在行）

    Returns:
        长格式 DataFrame，index 为 (time, id)，columns 为 item

    Example:
        >>> data = {f"factor_{i}": df for i, df in enumerate(dfs)}
        >>> result = panel_to_dataframe(data)
    """
    if isinstance(panel, dict):
        dfs = []
        for name, df in panel.items():
            if isinstance(df, pd.DataFrame):
                df = df.copy()
                df.columns = [name] * len(df.columns)
                dfs.append(df.stack().swaplevel())
            elif isinstance(df, pd.Series):
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, axis=1)
        result.columns = [c[1] for c in result.columns]
    elif isinstance(panel, list):
        if not panel:
            return pd.DataFrame()
        first = panel[0]
        if isinstance(first, pd.DataFrame):
            dfs = [df.stack() for df in panel]
            result = pd.concat(dfs, axis=1)
        else:
            result = pd.concat(panel, axis=1)
    else:
        raise TypeError(f"panel must be dict or list, got {type(panel)}")

    if orientation == "ftime":
        result = result.sort_index()
    return result


def dataframe_to_panel(
    df: pd.DataFrame,
    level: Union[str, int] = -1
) -> dict:
    """
    将宽格式 DataFrame 转换为 Panel（字典）

    Args:
        df: 宽格式 DataFrame，index 为时间，columns 为 item:id
        level: MultiIndex 的层级用于分割成 panel items

    Returns:
        字典 {item: DataFrame}，每个 DataFrame 的 index 为时间，columns 为 id

    Example:
        >>> df = pd.DataFrame(...)
        >>> panel = dataframe_to_panel(df, level=0)
    """
    if isinstance(df.columns, pd.MultiIndex):
        items = df.columns.get_level_values(level).unique()
        panel = {}
        for item in items:
            mask = df.columns.get_level_values(level) == item
            panel[item] = df.loc[:, mask].droplevel(level, axis=1)
        return panel
    else:
        return {"_default": df}


def align_dataframes(
    dfs: List[pd.DataFrame],
    how: str = "outer",
    axis: int = 0
) -> List[pd.DataFrame]:
    """
    对齐多个 DataFrame 的索引

    Args:
        dfs: DataFrame 列表
        how: 对齐方式 ("inner", "outer", "left", "right")
        axis: 对齐轴 (0=行索引, 1=列)

    Returns:
        对齐后的 DataFrame 列表

    Example:
        >>> aligned = align_dataframes([df1, df2, df3])
    """
    if not dfs:
        return []
    if len(dfs) == 1:
        return dfs

    if axis == 0:
        if how == "outer":
            index = dfs[0].index
            for df in dfs[1:]:
                index = index.union(df.index)
        elif how == "inner":
            index = dfs[0].index
            for df in dfs[1:]:
                index = index.intersection(df.index)
        elif how == "left":
            index = dfs[0].index
        elif how == "right":
            index = dfs[-1].index
        else:
            raise ValueError(f"how must be one of inner/outer/left/right, got {how}")

        return [df.reindex(index) for df in dfs]
    else:
        if how == "outer":
            columns = dfs[0].columns
            for df in dfs[1:]:
                columns = columns.union(df.columns)
        elif how == "inner":
            columns = dfs[0].columns
            for df in dfs[1:]:
                columns = columns.intersection(df.columns)
        elif how == "left":
            columns = dfs[0].columns
        elif how == "right":
            columns = dfs[-1].columns

        return [df.reindex(columns, axis=1) for df in dfs]


def forward_fill_panel(
    panel: dict,
    limit: Optional[int] = None
) -> dict:
    """
    对 Panel 中的每个 DataFrame 执行前向填充

    Args:
        panel: Panel 字典 {item: DataFrame}
        limit: 最大填充期数

    Returns:
        填充后的 Panel

    Example:
        >>> filled = forward_fill_panel(panel, limit=5)
    """
    result = {}
    for name, df in panel.items():
        if isinstance(df, pd.DataFrame):
            result[name] = df.ffill(limit=limit)
        else:
            result[name] = df
    return result


def fillna_by_value(
    df: pd.DataFrame,
    value: Union[float, dict],
    condition: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    按条件填充 NaN 值

    Args:
        df: 输入 DataFrame
        value: 填充值，如果是字典则为 {column: value}
        condition: 条件 DataFrame，True 的位置才填充

    Returns:
        填充后的 DataFrame

    Example:
        >>> df = fillna_by_value(df, 0)
        >>> df = fillna_by_value(df, {"col1": -999, "col2": 0})
    """
    result = df.copy()
    if isinstance(value, dict):
        for col, val in value.items():
            if col in result.columns:
                mask = result[col].isna()
                if condition is not None:
                    mask = mask & condition[col]
                result.loc[mask, col] = val
    else:
        mask = result.isna()
        if condition is not None:
            mask = mask & condition
        result = result.fillna(value)
        result = result.where(~mask, other=value)
    return result


def winsorize_series(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.01
) -> pd.Series:
    """
    对 Series 进行 Winsorize 去极值处理

    Args:
        s: 输入 Series
        lower: 下界百分位
        upper: 上界百分位

    Returns:
        去极值后的 Series

    Example:
        >>> s = winsorize_series(factor, lower=0.02, upper=0.02)
    """
    if len(s) == 0:
        return s

    lower_val = np.nanpercentile(s, lower * 100)
    upper_val = np.nanpercentile(s, (1 - upper) * 100)

    result = s.clip(lower=lower_val, upper=upper_val)
    return result


def standardize_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ddof: int = 1
) -> pd.DataFrame:
    """
    Z-Score 标准化

    Args:
        df: 输入 DataFrame
        columns: 要标准化的列，None 表示全部
        ddof: 标准差自由度

    Returns:
        标准化后的 DataFrame

    Example:
        >>> df = standardize_zscore(factor_df)
    """
    result = df.copy()
    cols = columns if columns is not None else df.columns

    for col in cols:
        if col in result.columns:
            mean = result[col].mean()
            std = result[col].std(ddof=ddof)
            if std > 0:
                result[col] = (result[col] - mean) / std
            else:
                result[col] = 0

    return result


def standardize_rank(
    s: pd.Series,
    ascending: bool = True,
    pct: bool = True
) -> pd.Series:
    """
    Rank 标准化

    Args:
        s: 输入 Series
        ascending: 是否升序
        pct: 是否返回百分位 (0-1)

    Returns:
        Rank 标准化后的 Series

    Example:
        >>> rank_factor = standardize_rank(factor)
    """
    if len(s) == 0:
        return s

    rank = s.rank(ascending=ascending, pct=pct)
    return rank


def cross_section_zscore(
    df: pd.DataFrame,
    groupby: Optional[Any] = None
) -> pd.DataFrame:
    """
    截面 Z-Score 标准化

    对每个时间点的截面数据进行标准化

    Args:
        df: 输入 DataFrame，index 为时间，columns 为 ID
        groupby: 分组列（如果有）

    Returns:
        标准化后的 DataFrame

    Example:
        >>> zscored = cross_section_zscore(factor_df)
    """
    if groupby is None:
        return df.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean(), axis=1)
    else:
        return df.groupby(groupby).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
        )


def cross_section_rank(
    df: pd.DataFrame,
    ascending: bool = False,
    pct: bool = True
) -> pd.DataFrame:
    """
    截面排名标准化

    Args:
        df: 输入 DataFrame
        ascending: 是否升序
        pct: 是否返回百分位

    Returns:
        排名标准化后的 DataFrame

    Example:
        >>> ranked = cross_section_rank(factor_df)
    """
    return df.rank(ascending=ascending, pct=pct, axis=1)


def shift_df(
    df: pd.DataFrame,
    periods: int,
    freq: Optional[str] = None
) -> pd.DataFrame:
    """
    移动 DataFrame（支持按频率）

    Args:
        df: 输入 DataFrame
        periods: 移动期数
        freq: 频率字符串（如 "D", "M"）

    Returns:
        移动后的 DataFrame

    Example:
        >>> shifted = shift_df(factor_df, periods=1)
    """
    if freq is None:
        return df.shift(periods)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("freq requires DatetimeIndex")
        new_index = df.index.shift(periods, freq=freq)
        return df.set_index(new_index)


def resample_panel(
    panel: dict,
    rule: str,
    agg_func: str = "last"
) -> dict:
    """
    对 Panel 按规则重采样

    Args:
        panel: Panel 字典
        rule: 重采样规则（如 "M", "Q", "Y"）
        agg_func: 聚合函数

    Returns:
        重采样后的 Panel

    Example:
        >>> monthly = resample_panel(daily_panel, rule="M")
    """
    result = {}
    for name, df in panel.items():
        if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
            result[name] = df.resample(rule).agg(agg_func)
        else:
            result[name] = df
    return result


def melt_panel(
    panel: dict,
    var_name: str = "item",
    value_name: str = "value"
) -> pd.DataFrame:
    """
    将 Panel 转换为长格式

    Args:
        panel: Panel 字典
        var_name: 变量名列名
        value_name: 值列名

    Returns:
        长格式 DataFrame

    Example:
        >>> long_df = melt_panel(panel)
    """
    dfs = []
    for name, df in panel.items():
        if isinstance(df, pd.DataFrame):
            temp = df.copy()
            temp[var_name] = name
            dfs.append(temp.reset_index())
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def pivot_long(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: List[str],
    var_name: str = "variable",
    value_name: str = "value"
) -> pd.DataFrame:
    """
    将宽格式 DataFrame 转换为长格式（类似 pd.melt）

    Args:
        df: 输入 DataFrame
        id_vars: ID 列
        value_vars: 值列
        var_name: 变量列名
        value_name: 值列名

    Returns:
        长格式 DataFrame
    """
    return pd.melt(
        df, id_vars=id_vars, value_vars=value_vars,
        var_name=var_name, value_name=value_name,
    )


def pivot_wide(
    df: pd.DataFrame,
    index: Union[str, List[str]],
    columns: str,
    values: str,
    aggfunc: str = "last"
) -> pd.DataFrame:
    """
    将长格式 DataFrame 转换为宽格式（类似 pd.pivot）

    Args:
        df: 输入 DataFrame
        index: 索引列
        columns: 列名来源列
        values: 值来源列
        aggfunc: 聚合函数

    Returns:
        宽格式 DataFrame
    """
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)


__all__ = [
    "panel_to_dataframe",
    "dataframe_to_panel",
    "align_dataframes",
    "forward_fill_panel",
    "fillna_by_value",
    "winsorize_series",
    "standardize_zscore",
    "standardize_rank",
    "cross_section_zscore",
    "cross_section_rank",
    "shift_df",
    "resample_panel",
    "melt_panel",
    "pivot_long",
    "pivot_wide",
]
