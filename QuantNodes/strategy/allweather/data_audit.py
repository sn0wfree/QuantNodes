"""通用回测数据对齐审计 (§5.33).

非修复性诊断脚本: 检查 panel_v3 与底层数据源对齐情况,
输出结构化报告 (CSV + 控制台表格), 不修改任何数据.

用法:
    python -m QuantNodes.strategy.allweather.data_audit
    # 或:
    from QuantNodes.strategy.allweather.data_audit import DataAuditor
    auditor = DataAuditor(panel_path="data/allweather/panel_v3.parquet")
    auditor.run_all()
"""
from __future__ import annotations

import duckdb
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    DUCKDB_PATH,
    ETF_NAV_PATH,
    ETF_SMARTBETA_PATH,
    ETF_OHLCV_ADJ_PATH,
)


class DataAuditor:
    """回测数据对齐审计 (§5.33).

    步骤:
    1. audit_panel_columns()     — panel 每列的完整度, 首末日期, 非空率
    2. audit_duckdb_futures()    — DuckDB continuous_main_* vs panel
    3. audit_etf_nav()           — NAV parquet 各 code vs panel
    4. audit_ohlcv()             — OHLCV 复权面板 vs NAV vs panel
    5. audit_index_sources()     — v7_index_prices (中证1000/债券)
    6. generate_report()         — 结构化输出 (csv + 控制台)
    """

    FUT_TABLE = "continuous_main_{p}_daily"
    COLS_COL_DATE = "trade_date"
    PANEL_END = pd.Timestamp("2026-07-21")
    PANEL_START = pd.Timestamp("2018-01-08")
    MIN_NONNULL_RATE = 0.95

    def __init__(self,
                 panel_path: str = "data/allweather/panel_v3.parquet",
                 output_dir: str = "data/allweather/audit"):
        self.panel_path = Path(panel_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.panel: Optional[pd.DataFrame] = None
        self.findings: list[dict] = []
        self.results: dict[str, pd.DataFrame] = {}

    def load_panel(self) -> pd.DataFrame:
        if self.panel is None:
            self.panel = pd.read_parquet(self.panel_path)
        return self.panel

    # ── 1. panel 列完整性 ──
    def audit_panel_columns(self) -> pd.DataFrame:
        panel = self.load_panel()
        records = []
        for col in panel.columns:
            s = panel[col].dropna()
            n_valid = len(s)
            rate = n_valid / len(panel)
            if n_valid == 0:
                first, last, span = None, None, 0
                status = "EMPTY"
            else:
                first = s.index.min()
                last = s.index.max()
                span = (last - first).days
                truncated = last < self.PANEL_END - pd.Timedelta(days=14)
                delayed = first > self.PANEL_START + pd.Timedelta(days=180)
                if truncated and delayed:
                    status = "TRUNCATED+LATE_START"
                elif truncated:
                    status = "TRUNCATED"
                elif delayed:
                    status = "LATE_START"
                elif rate < self.MIN_NONNULL_RATE:
                    status = "GAPS"
                else:
                    status = "OK"
            records.append({
                "column": col,
                "asset_class": self._asset_class(col),
                "first_date": first.date() if first else None,
                "last_date": last.date() if last else None,
                "span_days": span,
                "n_valid": n_valid,
                "non_null_rate": f"{rate:.2%}",
                "status": status,
            })
            if status != "OK":
                self.findings.append({
                    "category": "panel_column",
                    "column": col,
                    "status": status,
                    "first_date": str(first.date()) if first else None,
                    "last_date": str(last.date()) if last else None,
                    "non_null_rate": f"{rate:.2%}",
                    "detail": self._column_detail(col, panel, status),
                })
        df = pd.DataFrame(records)
        self.results["panel_columns"] = df
        return df

    def _asset_class(self, col: str) -> str:
        if col.startswith("fut_t") or col.startswith("fut_tf"):
            return "bond_rate"
        if col in ("etf_513100", "etf_513500", "etf_159740"):
            return "equity_etf_off"
        if col in ("etf_512890", "etf_159915", "etf_510300", "csi_1000"):
            return "equity_etf_dom"
        if col == "fut_au":
            return "precious"
        if col == "fut_cu":
            return "nonferrous"
        if col == "fut_sc":
            return "energy_chem"
        if col == "fut_m":
            return "agri_oilseed"
        if col.startswith("fut_"):
            return "fut_other"
        return "unknown"

    def _column_detail(self, col: str, panel: pd.DataFrame, status: str) -> str:
        s = panel[col]
        if status == "EMPTY":
            return "列完全为空"
        if status == "TRUNCATED":
            last = s.dropna().index.max()
            gap = (self.PANEL_END - last).days
            return f"末日期距 panel 截止 {gap} 天"
        if status == "LATE_START":
            first = s.dropna().index.min()
            gap = (first - self.PANEL_START).days
            return f"首日期晚于 panel 起始 {gap} 天 (晚上市资产)"
        if status == "GAPS":
            return "非空率 < 95%, 内部有零值或缺失"
        return ""

    # ── 2. DuckDB 期货表 vs panel ──
    def audit_duckdb_futures(self) -> pd.DataFrame:
        panel = self.load_panel()
        fut_cols = [c for c in panel.columns if c.startswith("fut_")]
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        records = []
        try:
            for col in fut_cols:
                p = col.replace("fut_", "").lower()
                table = self.FUT_TABLE.format(p=p)
                try:
                    df = con.execute(
                        f"SELECT MIN({self.COLS_COL_DATE}), MAX({self.COLS_COL_DATE}), "
                        f"COUNT(*) FROM {table}"
                    ).df()
                    duck_min = pd.to_datetime(df.iloc[0, 0])
                    duck_max = pd.to_datetime(df.iloc[0, 1])
                    duck_n = int(df.iloc[0, 2])
                    panel_s = panel[col].dropna()
                    p_min = panel_s.index.min()
                    p_max = panel_s.index.max()
                    p_n = len(panel_s)
                    # 一致性
                    date_match = abs((duck_max - p_max).days) <= 14
                    if not date_match:
                        self.findings.append({
                            "category": "duckdb_mismatch",
                            "column": col,
                            "duckdb_table": table,
                            "duckdb_max": str(duck_max.date()),
                            "panel_max": str(p_max.date()),
                            "status": "DATE_MISMATCH",
                            "detail": f"末日期差 {(duck_max - p_max).days} 天",
                        })
                    if duck_n > p_n * 1.1:
                        self.findings.append({
                            "category": "duckdb_row_mismatch",
                            "column": col,
                            "duckdb_rows": duck_n,
                    "panel_rows": p_n,
                    "status": "DUCKDB_HAS_MORE",
                    "detail": (f"DuckDB {duck_n} 行 > panel {p_n} 行"
                                f" (可能 panel 加载时丢行)"),
                        })
                    records.append({
                        "column": col,
                        "duckdb_table": table,
                        "duckdb_first": duck_min.date(),
                        "duckdb_last": duck_max.date(),
                        "duckdb_rows": duck_n,
                        "panel_first": p_min.date(),
                        "panel_last": p_max.date(),
                        "panel_rows": p_n,
                        "date_match": date_match,
                        "row_ratio": f"{duck_n/p_n:.2f}" if p_n > 0 else "N/A",
                    })
                except Exception as e:
                    records.append({
                        "column": col,
                        "duckdb_table": table,
                        "duckdb_first": None,
                        "duckdb_last": None,
                        "duckdb_rows": 0,
                        "panel_first": None,
                        "panel_last": None,
                        "panel_rows": 0,
                        "date_match": False,
                        "row_ratio": "N/A",
                    })
                    self.findings.append({
                        "category": "duckdb_missing",
                        "column": col,
                        "duckdb_table": table,
                        "status": "TABLE_NOT_FOUND",
                        "detail": f"查询失败: {e}",
                    })
        finally:
            con.close()
        df = pd.DataFrame(records)
        self.results["duckdb_futures"] = df
        return df

    # ── 3. NAV parquet vs panel ──
    def audit_etf_nav(self) -> pd.DataFrame:
        panel = self.load_panel()
        etf_cols = [c for c in panel.columns if c.startswith("etf_")]
        try:
            nav_main = pd.read_parquet(ETF_NAV_PATH)
        except Exception as e:
            self.findings.append({"category": "nav_load_failed",
                                  "path": str(ETF_NAV_PATH),
                                  "status": "FAILED", "detail": str(e)})
            return pd.DataFrame()
        try:
            nav_smart = pd.read_parquet(ETF_SMARTBETA_PATH)
        except Exception:
            nav_smart = pd.DataFrame()
        records = []
        for col in etf_cols:
            code = col.replace("etf_", "")
            s = panel[col].dropna()
            p_first = s.index.min() if len(s) else None
            p_last = s.index.max() if len(s) else None
            nav_first = nav_last = None
            source = "nav_main"
            if code in nav_main.columns:
                nav_s = nav_main[code].dropna()
                if len(nav_s):
                    nav_first = nav_s.index.min()
                    nav_last = nav_s.index.max()
            elif code in nav_smart.columns:
                nav_s = nav_smart[code].dropna()
                if len(nav_s):
                    nav_first = nav_s.index.min()
                    nav_last = nav_s.index.max()
                source = "nav_smartbeta"
            else:
                source = "MISSING"
            date_match = (nav_last is not None and p_last is not None
                          and abs((nav_last - p_last).days) <= 14)
            if not date_match and nav_last is not None and p_last is not None:
                self.findings.append({
                    "category": "nav_mismatch",
                    "column": col,
                    "source": source,
                    "nav_last": str(nav_last.date()) if nav_last else None,
                    "panel_last": str(p_last.date()) if p_last else None,
                    "status": "DATE_MISMATCH",
                    "detail": (f"末日期差 {(nav_last - p_last).days}"
                                f" if nav_last and p_last else 'N/A' 天"),
                })
            records.append({
                "column": col,
                "code": code,
                "source": source,
                "nav_first": nav_first.date() if nav_first is not None else None,
                "nav_last": nav_last.date() if nav_last is not None else None,
                "panel_first": p_first.date() if p_first else None,
                "panel_last": p_last.date() if p_last else None,
                "date_match": date_match,
            })
        df = pd.DataFrame(records)
        self.results["etf_nav"] = df
        return df

    # ── 4. OHLCV vs NAV vs panel ──
    def audit_ohlcv(self) -> pd.DataFrame:
        panel = self.load_panel()
        etf_cols = [c for c in panel.columns if c.startswith("etf_")]
        try:
            ohlcv = pd.read_parquet(ETF_OHLCV_ADJ_PATH)
        except Exception as e:
            self.findings.append({"category": "ohlcv_load_failed",
                                  "path": str(ETF_OHLCV_ADJ_PATH),
                                  "status": "FAILED", "detail": str(e)})
            return pd.DataFrame()
        # 资产代码在列 MultiIndex (code, field) 的 level 0; 旧代码误用 index
        # (DatetimeIndex) → 得到日期集合 → in_ohlcv 全部误报 False (§5.44 修复)
        if isinstance(ohlcv.columns, pd.MultiIndex):
            available_codes = {str(c) for c in ohlcv.columns.get_level_values(0)}
        else:
            available_codes = {str(c) for c in ohlcv.columns}
        records = []
        for col in etf_cols:
            code = col.replace("etf_", "")
            in_ohlcv = code in available_codes
            if in_ohlcv:
                close_s = ohlcv[(code, "close")].dropna()
                if len(close_s):
                    ohlcv_first = close_s.index.min()
                    ohlcv_last = close_s.index.max()
                else:
                    ohlcv_first = ohlcv_last = None
            else:
                ohlcv_first = ohlcv_last = None
            panel_s = panel[col].dropna()
            p_first = panel_s.index.min() if len(panel_s) else None
            p_last = panel_s.index.max() if len(panel_s) else None
            if in_ohlcv and ohlcv_last and p_last:
                gap = (ohlcv_last - p_last).days
                if abs(gap) > 14:
                    self.findings.append({
                        "category": "ohlcv_mismatch",
                        "column": col,
                        "ohlcv_last": str(ohlcv_last.date()),
                        "panel_last": str(p_last.date()),
                        "status": "DATE_MISMATCH",
                        "detail": f"OHLCV 末日期差 panel {gap} 天",
                    })
            records.append({
                "column": col,
                "code": code,
                "in_ohlcv": in_ohlcv,
                "ohlcv_first": ohlcv_first.date() if ohlcv_first is not None else None,
                "ohlcv_last": ohlcv_last.date() if ohlcv_last is not None else None,
                "panel_first": p_first.date() if p_first else None,
                "panel_last": p_last.date() if p_last else None,
            })
        df = pd.DataFrame(records)
        self.results["ohlcv"] = df
        return df

    # ── 5. 指数数据源 (v7_index_prices) ──
    def audit_index_sources(self) -> pd.DataFrame:
        try:
            idx = pd.read_parquet("data/high_freq_macro/v7_index_prices.parquet")
        except Exception as e:
            self.findings.append({"category": "index_load_failed",
                                  "path": "v7_index_prices",
                                  "status": "FAILED", "detail": str(e)})
            return pd.DataFrame()
        records = []
        panel_end = self.PANEL_END
        for col in idx.columns:
            s = idx[col].dropna()
            if len(s) == 0:
                continue
            last = s.index.max()
            gap = (panel_end - last).days
            status = "OK" if gap <= 14 else f"TRUNCATED_{gap}d"
            if gap > 14:
                self.findings.append({
                    "category": "index_truncated",
                    "column": col,
                    "last_date": str(last.date()),
                    "panel_end": str(panel_end.date()),
                    "status": status,
                    "detail": f"末日期距 panel 截止 {gap} 天",
                })
            records.append({
                "column": col,
                "first_date": s.index.min().date(),
                "last_date": last.date(),
                "n_rows": len(s),
                "gap_days_to_panel_end": gap,
                "status": status,
            })
        df = pd.DataFrame(records)
        self.results["index_sources"] = df
        return df

    # ── 报告输出 ──
    def generate_report(self, save_csv: bool = True) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("回测数据对齐审计报告 (§5.33)")
        lines.append(f"panel: {self.panel_path}")
        lines.append(f"panel 截止日: {self.PANEL_END.date()}")
        lines.append(f"panel 起始日: {self.PANEL_START.date()}")
        lines.append("=" * 70)

        lines.append("")
        lines.append(f"## 发现 (findings) — 共 {len(self.findings)} 项异常")
        lines.append("")
        if self.findings:
            by_cat = {}
            for f in self.findings:
                cat = f["category"]
                by_cat.setdefault(cat, []).append(f)
            for cat in sorted(by_cat):
                lines.append(f"### {cat} ({len(by_cat[cat])} 项)")
                for f in by_cat[cat]:
                    lines.append(f"  - {f.get('column', '?')}: "
                                  f"{f.get('status', '?')} -- {f.get('detail', '')}")
                lines.append("")
        else:
            lines.append("  无异常发现")
            lines.append("")

        for k, v in self.results.items():
            if not v.empty:
                lines.append(f"## {k}")
                lines.append(v.to_string(index=False))
                lines.append("")

        report = "\n".join(lines)
        if save_csv:
            csv_path = self.output_dir / "audit_findings.csv"
            pd.DataFrame(self.findings).to_csv(csv_path, index=False)
            for k, v in self.results.items():
                v.to_csv(self.output_dir / f"audit_{k}.csv", index=False)
            report += f"\n\nCSV 输出: {self.output_dir}\n"
        return report

    def run_all(self) -> str:
        self.load_panel()
        self.audit_panel_columns()
        self.audit_duckdb_futures()
        self.audit_etf_nav()
        self.audit_ohlcv()
        self.audit_index_sources()
        return self.generate_report()


if __name__ == "__main__":
    auditor = DataAuditor()
    print(auditor.run_all())
