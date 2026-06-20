# coding: utf-8
"""IFinDDatabase - iFinD API 包装为 DataLoader 兼容接口

Drop-in replacement for DataLoader, backed by iFinD API.
所有 panel 数据返回 (dates × stocks) DataFrame, index=日期(int), columns=股票代码(str).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import ClassVar, Callable

from .fetcher import IFindFetcher


# ── Route Decorator (B5 refactor, 2026-06-19) ──────────────────

def register_route(filename: str, key: str) -> Callable:
    """装饰器: 把方法注册到 IFinDDatabase._ROUTE_TABLE[(filename, key)] = method 名.

    使用::

        @register_route("stk_daily.h5", "cp")
        def _get_prices(self) -> pd.DataFrame:
            ...

        # 一个方法可注册到多个 (filename, key):
        @register_route("stk_daily.h5", "trade_dt")
        @register_route("index_daily.h5", "trade_dt")
        def _get_trade_dt_raw(self) -> pd.DataFrame:
            ...

    装饰器给方法打 ``_routes`` (列表) 标记; 实际表填充在类定义后由
    ``_collect_routes(IFinDDatabase)`` 一次性扫描完成, 保证
    ``load_h5`` 路由查找零开销.
    """
    def deco(method: Callable) -> Callable:
        existing = getattr(method, "_routes", None)  # type: ignore[attr-defined]
        if existing is None:
            method._routes = [(filename, key)]  # type: ignore[attr-defined]
        else:
            method._routes = existing + [(filename, key)]  # type: ignore[attr-defined]
        return method
    return deco


def _collect_routes(cls: type) -> dict[tuple[str, str], str]:
    """扫描类及所有父类, 收集 ``_routes`` 标记的方法.

    Returns:
        ``{(filename, key): method_name}`` 字典.
    """
    routes: dict[tuple[str, str], str] = {}
    for klass in reversed(cls.__mro__):
        for name, val in klass.__dict__.items():
            if callable(val) and hasattr(val, "_routes"):
                for route_key in val._routes:  # type: ignore[attr-defined]
                    routes[route_key] = name
    return routes


# ── Week 12: H5 兼容辅助 ─────────────────────────────────────

def _df_to_hdf_safe(df: pd.DataFrame) -> pd.DataFrame:
    """转 nullable/Int64 等 HDF5 不支持的 dtype 为兼容 dtype。

    - Int64 → int64 (含 NaN 时会丢失 NaN, 但 HDF5 不支持 nullable int)
    - int64 → int64
    - 其余保持
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'Int64':
            # 全部 NaN → float (否则 .astype(int) 报错)
            if df[col].isna().all():
                df[col] = df[col].astype(float)
            else:
                df[col] = df[col].fillna(0).astype(int)
    if df.index.dtype == 'Int64':
        if df.index.isna().all():
            df.index = df.index.astype(float)
        else:
            df.index = df.index.fillna(0).astype(int)
    return df


# ── 行业代码映射 ──────────────────────────────────────────────
# H13: 默认 30 申万一级, 可通过 constructor 的 industry_map 参数覆盖
_DEFAULT_INDUSTRY_MAP = {
    '农林牧渔': 1, '基础化工': 2, '钢铁': 3, '有色金属': 4, '电子': 5,
    '汽车': 6, '家用电器': 7, '食品饮料': 8, '纺织服饰': 9, '轻工制造': 10,
    '医药生物': 11, '公用事业': 12, '交通运输': 13, '房地产': 14, '商贸零售': 15,
    '社会服务': 16, '银行': 17, '非银金融': 18, '综合': 19, '建筑材料': 20,
    '建筑装饰': 21, '电力设备': 22, '国防军工': 23, '计算机': 24, '传媒': 25,
    '通信': 26, '煤炭': 27, '石油石化': 28, '环保': 29, '美容护理': 30,
}


class IFinDDatabase:
    """iFinD API-backed DataLoader replacement.

    Usage:
        db = IFinDDatabase(date_beg='20260101', date_end='20260630')
        # 与 DataLoader 完全兼容
        stklist, trade_dt = db.get_stock_axis()
        cp = db.load_h5('stk_daily.h5', 'cp')
        cp_labeled = db.add_index(cp)
    """

    def __init__(self, api_path: str = '', date_beg: str = '',
                 date_end: str = '', universe: str = '沪深300',
                 fetcher: IFindFetcher = None,
                 industry_map: dict | None = None,
                 risk_registry: list[str] | None = None,
                 batch_size: int | None = None):
        """
        Args:
            api_path: 兼容 DataLoader, 被忽略
            date_beg: 查询起始日期 (YYYYMMDD, 空=1 年前)
            date_end: 查询截止日期 (空=今天)
            universe: 股票池 ('沪深300', '中证500', 'all')
            fetcher: 注入的 fetcher (测试用 IFindFetcherStub)
            industry_map: H13 行业代码映射 (None=30 申万一级)
            risk_registry: P-4 风险因子注册表 (None=10 默认 Barra 风格因子)
            batch_size: M10 分批查询大小 (None=默认 50, 推荐 50-100 避免限流)
        """
        # H9: 不再硬编码 '20260101', 默认 1 年前 (跨年/跨月可滚动)
        if not date_beg:
            one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            self._date_beg = one_year_ago
        else:
            self._date_beg = date_beg
        self._date_end = date_end or datetime.now().strftime('%Y%m%d')
        self._universe = universe
        self._fetcher = fetcher or IFindFetcher()
        # H13: 行业代码映射可覆盖
        self._industry_map = industry_map if industry_map is not None else _DEFAULT_INDUSTRY_MAP
        # P-4: 风险因子注册表可外部注入, None=10 默认 Barra 风格因子
        self._risk_registry = list(risk_registry) if risk_registry is not None else [
            '/beta', '/momentum', '/size', '/volatility',
            '/value', '/quality', '/growth', '/leverage',
            '/liquidity', '/non_linear_size',
        ]
        # M10: 分批查询大小 (默认 50, 可自定义)
        self._batch_size = batch_size if batch_size is not None else 50

        # 缓存
        self._stklist = None
        self._indexlist = None
        self._trade_dt = None
        self._stock_prices = None
        self._index_prices = None
        self._stock_info_cache = {}  # key -> DataFrame

    # ── 路由表 (B5: 由 @register_route 装饰器收集, 类定义后一次性填充) ──
    #
    # 原 _ROUTE_TABLE 字面量已迁移到各 _get_* 方法上的 @register_route 装饰器,
    # 见类下方 _collect_routes(IFinDDatabase) 调用.
    # 旧 _ROUTE_TABLE 访问完全兼容 (dict 接口不变).
    _ROUTE_TABLE: ClassVar[dict[tuple[str, str], str]] = {}

    # ── DataLoader 兼容接口 ─────────────────────────────────

    def load_h5(self, filename: str, key: str) -> pd.DataFrame:
        """路由到 iFinD 查询, 模拟 H5 加载"""
        route_key = (filename, key)
        if route_key in self._ROUTE_TABLE:
            method_name = self._ROUTE_TABLE[route_key]
            return getattr(self, method_name)()
        raise KeyError(
            f"IFinDDatabase: 未映射的 (filename='{filename}', key='{key}'). "
            f"可用路由: {list(self._ROUTE_TABLE.keys())}"
        )

    def load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=0)

    def load_npy(self, path: str) -> pd.DataFrame:
        return pd.DataFrame(np.load(path, allow_pickle=True))

    def load_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def load_custom(self, data_dir: tuple) -> pd.DataFrame:
        raise NotImplementedError(
            "IFinDDatabase 不支持自定义路径加载, "
            "请使用 load_h5 或直接通过 fetcher.query()"
        )

    def load_factor(self, factor_dir: str, factor_name: str) -> pd.DataFrame:
        """因子加载: 从 iFinD 获取因子数据"""
        return self._get_factor(factor_dir, factor_name)

    def get_stock_axis(self) -> tuple:
        """返回 (stklist, trade_dt), 格式与 DataLoader 一致"""
        return self._get_stock_axis_raw(), self._get_trade_dt_raw()

    def get_index_axis(self) -> tuple:
        return self._get_index_axis_raw(), self._get_trade_dt_raw()

    def get_axis(self, axis_type: str = 'stock') -> tuple:
        if axis_type == 'stock':
            return self.get_stock_axis()
        elif axis_type == 'index':
            return self.get_index_axis()
        else:
            raise ValueError(f"不支持的 axis_type: {axis_type}")

    def add_index(self, factor: pd.DataFrame, axis_type: str = 'stock') -> pd.DataFrame:
        """给因子添加标准索引。iFinD 数据通常已带标签, 仅做验证"""
        factor = factor.copy()
        assetlist, trade_dt = self.get_axis(axis_type)

        # 如果已有正确维度但缺少标签, 则添加
        expected_dates = trade_dt.iloc[:, 0].values
        expected_assets = assetlist.iloc[:, 0].values

        if factor.shape == (len(expected_dates), len(expected_assets)):
            if not factor.index.equals(pd.Index(expected_dates)):
                factor.index = expected_dates
            if not factor.columns.equals(pd.Index(expected_assets)):
                factor.columns = expected_assets
        return factor

    def valid_shape(self, factor: pd.DataFrame, axis_type: str = 'stock') -> bool:
        assetlist, trade_dt = self.get_axis(axis_type)
        return factor.shape == (len(trade_dt), len(assetlist))

    def get_apikeys(self, filename: str) -> list:
        """风险因子注册表 (P-4: 改为读 self._risk_registry, 可外部注入)"""
        return list(self._risk_registry)

    # ── 内部数据获取方法 ─────────────────────────────────────

    def _query_stock_info(self, query: str) -> pd.DataFrame:
        """封装股票信息查询"""
        return self._fetcher.query('stock', 'get_stock_info', {'query': query})

    def _query_index_data(self, query: str) -> pd.DataFrame:
        """封装指数数据查询"""
        return self._fetcher.query('index', 'index_data', {'query': query})

    def _get_stock_codes(self) -> list[str]:
        """获取股票池代码列表"""
        if self._stklist is not None:
            return list(self._stklist.iloc[:, 0])

        if self._universe == 'all':
            query = f'A股市场所有股票代码({self._date_beg[:4]}年)'
        else:
            query = f'{self._universe}成分股列表'

        df = self._query_index_data(query)
        if df.empty:
            raise RuntimeError(f"无法获取股票池: {self._universe}")

        # 找到代码列
        code_col = None
        for col in df.columns:
            if df[col].astype(str).str.match(r'\d{6}\.(SH|SZ)').any():
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]

        codes = df[code_col].astype(str).tolist()
        self._stklist = pd.DataFrame(codes)
        return codes

    def _get_trade_dates(self) -> list[int]:
        """获取交易日历"""
        if self._trade_dt is not None:
            return list(self._trade_dt.iloc[:, 0])

        # 尝试从指数数据获取交易日
        try:
            beg_year = self._date_beg[:4]
            beg_month = self._date_beg[4:6]
            end_month = self._date_end[4:6]
            query = (
                f'沪深300、中证500{beg_year}年{beg_month}月至{end_month}月的收盘点数'
            )
            df = self._query_index_data(query)
            if not df.empty:
                date_col = None
                for col in df.columns:
                    if df[col].astype(str).str.match(r'^\d{8}$').any():
                        date_col = col
                        break
                if date_col:
                    dates = sorted(
                        pd.to_numeric(df[date_col], errors='coerce')
                        .dropna().astype(int).unique().tolist()
                    )
                    self._trade_dt = pd.DataFrame(dates)
                    return dates
        except Exception:
            pass

        # fallback: 从价格数据中提取日期
        prices = self._get_prices()
        dates = sorted(prices.index.tolist())
        self._trade_dt = pd.DataFrame(dates)
        return dates

    @register_route("stk_daily.h5", "stklist")
    def _get_stock_axis_raw(self) -> pd.DataFrame:
        """stklist DataFrame: (N_stocks, 1)"""
        if self._stklist is None:
            self._get_stock_codes()
        return self._stklist

    @register_route("index_daily.h5", "indexlist")
    def _get_index_axis_raw(self) -> pd.DataFrame:
        """indexlist DataFrame: (N_indices, 1)"""
        if self._indexlist is None:
            query = '沪深300、中证500收盘点数'
            df = self._query_index_data(query)
            if df.empty:
                self._indexlist = pd.DataFrame(['000300.SH', '000905.SH'])
            else:
                code_col = df.columns[0]
                codes = df[code_col].unique().tolist()
                self._indexlist = pd.DataFrame(codes)
        return self._indexlist

    @register_route("stk_daily.h5", "trade_dt")
    @register_route("index_daily.h5", "trade_dt")
    def _get_trade_dt_raw(self) -> pd.DataFrame:
        """trade_dt DataFrame: (M_dates, 1)"""
        if self._trade_dt is None:
            self._get_trade_dates()
        return self._trade_dt

    @register_route("stk_daily.h5", "cp")
    def _get_prices(self) -> pd.DataFrame:
        """获取股票收盘价面板 (dates × stocks)"""
        if self._stock_prices is not None:
            return self._stock_prices

        codes = self._get_stock_codes()
        # 分批查询 (每批最多 self._batch_size 个代码, M10 参数化)
        all_dfs = []
        for i in range(0, len(codes), self._batch_size):
            batch = codes[i:i + self._batch_size]
            code_str = '、'.join(batch)
            beg_year = self._date_beg[:4]
            beg_month = self._date_beg[4:6]
            end_month = self._date_end[4:6]
            query = (
                f'{code_str}{beg_year}年{beg_month}月至{end_month}月的日收盘价'
            )
            df = self._query_stock_info(query)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            raise RuntimeError("无法获取股票价格数据")

        combined = pd.concat(all_dfs, ignore_index=True)
        self._stock_prices = self._pivot_prices(combined)
        return self._stock_prices

    def _pivot_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """将长格式价格表转为宽格式 (dates × stocks)"""
        # 找到日期列、代码列、价格列
        date_col = None
        code_col = None
        price_col = None

        for col in df.columns:
            sample = df[col].astype(str)
            if sample.str.match(r'^\d{8}$').any():
                date_col = col
            elif sample.str.match(r'\d{6}\.(SH|SZ)').any():
                code_col = col
            elif '收盘' in col or '价' in col:
                price_col = col

        if date_col is None or code_col is None or price_col is None:
            # fallback: 用前3列
            cols = df.columns.tolist()
            code_col = code_col or cols[0]
            date_col = date_col or cols[1] if len(cols) > 1 else cols[0]
            price_col = price_col or cols[2] if len(cols) > 2 else cols[1]

        pivot = df.pivot_table(
            index=date_col, columns=code_col, values=price_col,
            aggfunc='first'
        )
        pivot.index = pd.to_numeric(pivot.index, errors='coerce').astype('Int64')
        pivot.columns = pivot.columns.astype(str)
        return pivot.sort_index()

    def _get_stock_info_panel(self, key: str, query_template: str) -> pd.DataFrame:
        """通用股票信息面板获取"""
        if key in self._stock_info_cache:
            return self._stock_info_cache[key]

        codes = self._get_stock_codes()
        # M10: 分批查询用 self._batch_size
        all_dfs = []
        for i in range(0, len(codes), self._batch_size):
            batch = codes[i:i + self._batch_size]
            code_str = '、'.join(batch)
            query = query_template.format(codes=code_str, beg=self._date_beg, end=self._date_end)
            df = self._query_stock_info(query)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            # 返回空面板
            dates = self._get_trade_dates()
            return pd.DataFrame(0, index=dates, columns=codes)

        combined = pd.concat(all_dfs, ignore_index=True)
        panel = self._pivot_stock_info(combined, codes)
        self._stock_info_cache[key] = panel
        return panel

    def _pivot_stock_info(self, df: pd.DataFrame, codes: list) -> pd.DataFrame:
        """将长格式转为宽格式 (dates × stocks)"""
        date_col = code_col = value_col = None
        for col in df.columns:
            sample = df[col].astype(str)
            if sample.str.match(r'^\d{8}$').any():
                date_col = col
            elif sample.str.match(r'\d{6}\.(SH|SZ)').any():
                code_col = col
            else:
                if value_col is None:
                    value_col = col

        if date_col and code_col and value_col:
            pivot = df.pivot_table(
                index=date_col, columns=code_col, values=value_col,
                aggfunc='first'
            )
            pivot.index = pd.to_numeric(pivot.index, errors='coerce').astype('Int64')
            # 确保所有股票都在列中
            for c in codes:
                if c not in pivot.columns:
                    pivot[c] = 0
            return pivot[codes].sort_index()

        # fallback
        dates = self._get_trade_dates()
        return pd.DataFrame(0, index=dates, columns=codes)

    @register_route("stk_daily.h5", "id_citic1")
    def _get_industry(self) -> pd.DataFrame:
        """行业分类面板 (dates × stocks), 值为申万一级代码 (1-30)"""
        return self._get_stock_info_panel(
            'id_citic1',
            '{codes}的行业分类(申万一级)'
        ).applymap(lambda x: self._industry_map.get(str(x), 0) if pd.notna(x) else 0)

    @register_route("stk_daily.h5", "mv_float")
    def _get_market_value(self) -> pd.DataFrame:
        """流通市值面板 (dates × stocks)"""
        return self._get_stock_info_panel(
            'mv_float',
            '{codes}的流通市值'
        )

    @register_route("stk_daily.h5", "st")
    def _get_st_status(self) -> pd.DataFrame:
        """ST 状态面板 (dates × stocks), 值为 0/1"""
        return self._get_stock_info_panel(
            'st',
            '{codes}是否被ST处理'
        ).applymap(lambda x: 1 if str(x).strip() in ('是', 'True', '1', 'ST') else 0)

    @register_route("stk_daily.h5", "suspend")
    def _get_suspension(self) -> pd.DataFrame:
        """停牌状态面板 (dates × stocks), 值为 0/1"""
        return self._get_stock_info_panel(
            'suspend',
            '{codes}是否停牌'
        ).applymap(lambda x: 1 if str(x).strip() in ('是', 'True', '1', '停牌') else 0)

    @register_route("stk_daily.h5", "ud_limit")
    def _get_limit(self) -> pd.DataFrame:
        """涨跌停面板 (dates × stocks), 值为 0/1/-1"""
        return self._get_stock_info_panel(
            'ud_limit',
            '{codes}是否涨跌停'
        ).applymap(lambda x: 1 if '涨停' in str(x) else (-1 if '跌停' in str(x) else 0))

    @register_route("stk_daily.h5", "ipo_days")
    def _get_ipo_days(self) -> pd.DataFrame:
        """上市天数面板 (dates × stocks)"""
        panel = self._get_stock_info_panel(
            'ipo_days',
            '{codes}的上市日期'
        )
        # 将上市日期转为天数
        dates = panel.index
        for col in panel.columns:
            ipo_date = panel[col].iloc[0] if not panel[col].empty else 0
            try:
                ipo_dt = pd.to_datetime(str(int(ipo_date)), format='%Y%m%d')
                panel[col] = [(d - ipo_dt).days if pd.notna(d) else 9999
                              for d in pd.to_datetime(dates.astype(str), format='%Y%m%d')]
            except Exception:
                panel[col] = 500
        return panel

    @register_route("index_daily.h5", "index_cp")
    def _get_index_cp(self) -> pd.DataFrame:
        """指数收盘价面板 (dates × indices)"""
        if self._index_prices is not None:
            return self._index_prices

        indexlist = self._get_index_axis_raw()
        indices = indexlist.iloc[:, 0].tolist()
        code_str = '、'.join(indices)
        beg_year = self._date_beg[:4]
        beg_month = self._date_beg[4:6]
        end_month = self._date_end[4:6]
        query = (
            f'{code_str}{beg_year}年{beg_month}月至{end_month}月的收盘点数'
        )
        df = self._query_index_data(query)

        if df.empty:
            dates = self._get_trade_dates()
            self._index_prices = pd.DataFrame(0, index=dates, columns=indices)
        else:
            self._index_prices = self._pivot_prices(df)
        return self._index_prices

    @register_route("stk_daily.h5", "id_300")
    def _get_hs300_member(self) -> pd.DataFrame:
        """沪深300成分股面板 (dates × stocks), 值为 0/1"""
        return self._get_index_member_panel('000300.SH', '沪深300')

    @register_route("stk_daily.h5", "id_500")
    def _get_zz500_member(self) -> pd.DataFrame:
        """中证500成分股面板 (dates × stocks), 值为 0/1"""
        return self._get_index_member_panel('000905.SH', '中证500')

    def _get_index_member_panel(self, index_code: str, index_name: str) -> pd.DataFrame:
        """指数成分股面板"""
        dates = self._get_trade_dates()
        codes = self._get_stock_codes()

        # 查询成分股
        query = f'{index_name}成分股列表'
        df = self._query_index_data(query)

        member_codes = set()
        if not df.empty:
            for col in df.columns:
                vals = df[col].astype(str).str.strip()
                member_codes.update(vals[vals.str.match(r'\d{6}\.(SH|SZ)')].tolist())

        # 构建面板
        panel = pd.DataFrame(0, index=dates, columns=codes)
        for c in codes:
            if c in member_codes:
                panel[c] = 1
        return panel

    def _get_factor(self, factor_dir: str, factor_name: str) -> pd.DataFrame:
        """通过 iFinD 获取因子数据"""
        beg_year = self._date_beg[:4]
        beg_month = self._date_beg[4:6]
        end_month = self._date_end[4:6]
        query = (
            f'{factor_name}因子{beg_year}年{beg_month}月至{end_month}月'
        )
        df = self._query_stock_info(query)
        if df.empty:
            raise RuntimeError(f"无法获取因子: {factor_name}")
        codes = self._get_stock_codes()
        return self._pivot_stock_info(df, codes)

    # ── Week 12: 真实数据拉取 + H5 持久化 ────────────────────────

    def fetch_to_h5(
        self,
        output_dir: str | Path,
        factor_names: list[str] | None = None,
        keys: list[str] | None = None,
    ) -> dict:
        """从 iFinD 拉取数据, 写为 HDF5 格式 (兼容 LoadDataNode)。

        输出文件结构:
            {output_dir}/
            ├── stk_daily.h5         # 7 keys: cp, st, suspend, ud_limit,
            │                          #        ipo_days, id_citic1, mv_float
            ├── index_daily.h5       # 1 key: index_cp
            ├── stklist.h5           # 股票代码列表
            ├── trade_dt.h5          # 交易日历
            └── {factor_name}.h5     # 因子数据 (单 key='data')

        Args:
            output_dir: 输出目录 (不存在自动创建)
            factor_names: 因子名列表 (空=不拉因子)
            keys: stk_daily.h5 要拉的 key 列表 (默认全 7 个)

        Returns:
            dict: {file: {key: shape}, ...} 拉取统计
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_keys = ['cp', 'st', 'suspend', 'ud_limit', 'ipo_days', 'id_citic1', 'mv_float']
        if keys is None:
            keys = all_keys
        stats: dict = {}

        # 1. 拉股票池 + 交易日历
        print(f"[1/4] 拉股票池 ({self._universe})...")
        stklist = self._get_stock_axis_raw()
        trade_dt = self._get_trade_dt_raw()
        stklist.to_hdf(output_dir / 'stklist.h5', key='data', mode='w')
        trade_dt.to_hdf(output_dir / 'trade_dt.h5', key='data', mode='w')
        stats['stklist.h5'] = {'data': stklist.shape}
        stats['trade_dt.h5'] = {'data': trade_dt.shape}
        print(f"  ✓ stklist.h5 (shape={stklist.shape}), trade_dt.h5 (shape={trade_dt.shape})")

        # 2. 拉 stk_daily 7 keys
        print(f"[2/4] 拉 stk_daily.h5 (keys={keys})...")
        stk_daily_stats: dict = {}
        with pd.HDFStore(output_dir / 'stk_daily.h5', mode='w') as store:
            for key in keys:
                if key not in all_keys:
                    print(f"  ⚠ 跳过未知 key: {key}")
                    continue
                try:
                    df = self.load_h5('stk_daily.h5', key)
                    if df is None or df.empty:
                        print(f"  ⚠ {key}: 空数据, 跳过")
                        continue
                    # 转换 Int64 (nullable) → int64, HDF5 不支持 nullable int
                    df = _df_to_hdf_safe(df)
                    store.put(key, df, format='table')
                    stk_daily_stats[key] = df.shape
                    print(f"  ✓ {key}: shape={df.shape}")
                except Exception as e:
                    print(f"  ✗ {key} 失败: {e}")
        stats['stk_daily.h5'] = stk_daily_stats

        # 3. 拉 index_daily (沪深300 + 中证500)
        print("[3/4] 拉 index_daily.h5 (index_cp)...")
        try:
            index_cp = self._get_index_cp()
            with pd.HDFStore(output_dir / 'index_daily.h5', mode='w') as store:
                store.put('index_cp', _df_to_hdf_safe(index_cp), format='table')
            stats['index_daily.h5'] = {'index_cp': index_cp.shape}
            print(f"  ✓ index_cp: shape={index_cp.shape}")
        except Exception as e:
            print(f"  ✗ index_cp 失败: {e}")
            stats['index_daily.h5'] = {'index_cp': None}

        # 4. 拉因子 (可选)
        if factor_names:
            print(f"[4/4] 拉因子 ({factor_names})...")
            for fname in factor_names:
                try:
                    factor = self._get_factor(fname, fname)
                    factor = _df_to_hdf_safe(factor)
                    factor.to_hdf(output_dir / f'{fname}.h5', key='data', mode='w')
                    stats[f'{fname}.h5'] = {'data': factor.shape}
                    print(f"  ✓ {fname}: shape={factor.shape}")
                except Exception as e:
                    print(f"  ✗ {fname} 失败: {e}")
                    stats[f'{fname}.h5'] = {'data': None}
        else:
            print("[4/4] 跳过因子拉取 (factor_names 为空)")

        return stats

    def get_universe_stocks(self) -> list[str]:
        """获取股票池代码列表 (公开 API, 缓存)。"""
        return self._get_stock_codes()


# ── B5: 类定义后扫描所有 @register_route 标记, 一次性填充 _ROUTE_TABLE ──
IFinDDatabase._ROUTE_TABLE = _collect_routes(IFinDDatabase)
