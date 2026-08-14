# coding=utf-8
"""期权组合优化器 — 从 期权工作平台/op1.py 移植.

核心:
    - OptBundle: SLSQP 组合优化 (delta/gamma/vega 约束)
    - OptionPortfolioWithDT: 单时间截面组合容器
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

from .utils import cal_pnl, required_share, cal_portfolio_greek


class OptBundle:
    """SLSQP 期权组合优化器.

    约束:
        - delta_exposure = target (等式)
        - gamma = 0 (等式, delta-hedged gamma neutral)
        - vega >= 0 (不等式)
        - |sum(weights)| <= 50 (仓位约束)

    目标:
        min |cost| + (|delta_curve_stability| + |pnl_curve_stability|)^2
    """

    @staticmethod
    def delta_exposure_constraint(weights, Delta, f, target_delta_exposure):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f - target_delta_exposure

    @staticmethod
    def delta(weights, Delta, f):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f

    @staticmethod
    def gamma_constraint(weights, Gamma, f):
        return np.sum(weights * Gamma.loc[f, :] * 100)

    @staticmethod
    def vega_constraint(weights, Vega, f):
        return np.sum(weights * Vega.loc[f, :] * 100)

    @staticmethod
    def fee(weights, fee):
        return np.sum(weights * fee * 100) * -1

    @staticmethod
    def delta_curve_stability(weights, delta_matrix, f, price_range):
        merged = delta_matrix.reindex(index=price_range).dot(weights)
        return np.mean(np.power(np.gradient(merged, price_range - f), 2))

    @classmethod
    def pnl_stability_constraint(
        cls, lots, fee, f, selected, k_list, price_range, tolerance=10
    ):
        selected = selected.copy()
        selected["weight"] = np.int32(lots)
        pnls = pd.Series(
            {f_: cal_pnl(f_, selected) - fee for f_ in price_range}
        ).to_frame("pnl")
        sub_pnls = pnls[pnls.index.isin(price_range)]["pnl"]

        if isinstance(price_range, (list, np.ndarray)) and len(price_range) > 1:
            dx = np.diff(price_range).mean()
            stability_measure = np.abs(np.gradient(sub_pnls, dx))
        else:
            stability_measure = np.abs(np.gradient(sub_pnls, edge_order=1))

        return tolerance - np.max(stability_measure)

    @classmethod
    def objective(cls, weights, fee, delta, f, mask_selected, price_range):
        cost = cls.fee(weights, fee)
        delta_curve = cls.delta_curve_stability(weights, delta, f, price_range)
        pnl_curve = cls.pnl_stability_constraint(
            weights, cost, f, mask_selected, delta.index, price_range
        )
        return np.abs(cost) + (np.abs(delta_curve) + np.abs(pnl_curve)) ** 2

    @classmethod
    def run_opt(
        cls,
        initial_weights,
        mask_selected,
        target_delta_exposure,
        Delta,
        Gamma,
        Vega,
        Rho,
        method: str = "SLSQP",
        verbose: bool = True,
    ):
        """运行 SLSQP 优化.

        Args:
            initial_weights: 初始权重
            mask_selected: 筛选后的期权数据 (含 f/k/fee/cp 列)
            target_delta_exposure: 目标 delta 敞口
            Delta/Gamma/Vega/Rho: 矩阵 (index=strike, columns=contract)
            method: 优化方法
            verbose: 是否打印结果

        Returns:
            scipy.optimize.OptimizeResult
        """
        f = mask_selected["f"].unique().tolist()[0]
        price_range = Delta[
            (Delta.index <= f + 100) & (Delta.index >= f - 200)
        ].index
        fee = mask_selected["fee"]

        total_share_constraint = LinearConstraint(
            np.ones(len(fee)), lb=-50, ub=50
        )

        c1 = (
            {
                "type": "eq",
                "fun": cls.delta_exposure_constraint,
                "args": (Delta, f, target_delta_exposure),
            },
            {
                "type": "eq",
                "fun": cls.gamma_constraint,
                "args": (Gamma, f),
            },
            {
                "type": "ineq",
                "fun": cls.vega_constraint,
                "args": (Vega, f),
            },
            total_share_constraint,
        )

        objective = lambda w: cls.objective(w, fee, Delta, f, mask_selected, price_range)
        bounds = [(-50, 50)] * len(initial_weights)

        result = minimize(
            objective, initial_weights, constraints=c1, bounds=bounds, method=method
        )

        opt_weight = np.int32(result.x)

        if verbose:
            print(f"损失函数: {objective(opt_weight):.4f}")
            print(f"f: {f}")
            print(f"优化后权重: {opt_weight}")
            print(f"Delta: {cls.delta(opt_weight, Delta, f):.2f}")
            print(f"Gamma: {cls.gamma_constraint(opt_weight, Gamma, f):.4f}")
            print(f"Vega: {cls.vega_constraint(opt_weight, Vega, f):.2f}")
            print(f"Cost: {cls.fee(opt_weight, fee):.2f}")

        return result


class OptionPortfolioWithDT:
    """单时间截面期权组合容器.

    用于构建和分析某一时点的期权组合.
    """

    __slots__ = ("_selected", "_multiply_num", "_weight")

    def __init__(self, selected, multiply_num: int = 100):
        if len(selected["dt"].unique().tolist()) != 1:
            raise ValueError("Only cross-sectional data accepted (single dt)")
        self._selected = selected
        self._multiply_num = multiply_num
        self._weight = []

    @property
    def call(self):
        return self._selected[self._selected["cp"] == "C"]

    @property
    def put(self):
        return self._selected[self._selected["cp"] == "P"]

    def level_put(self, level: int = 1):
        put = self.put.sort_values("f_k_diff")
        otm = put[put["f_k_diff"] > 0]
        return otm.head(level)["contract_code"].values[-1] if len(otm) >= level else None

    def level_call(self, level: int = 1):
        call = self.call.sort_values("f_k_diff", ascending=False)
        otm = call[call["f_k_diff"] < 0]
        return otm.head(level)["contract_code"].values[-1] if len(otm) >= level else None

    def get_level_put(self, level: int = 1):
        code = self.level_put(level)
        return self.get(code) if code else None

    def get_level_call(self, level: int = 1):
        code = self.level_call(level)
        return self.get(code) if code else None

    def get(self, contract_code):
        return self._selected[self._selected["contract_code"] == contract_code]

    @property
    def f(self):
        return self._selected["f"].unique().tolist()[0]

    @property
    def available_contracts(self):
        return self._selected["contract_code"].unique().tolist()

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    def portfolio_delta(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected["Delta"]) * self._multiply_num * self.f

    def portfolio_gamma(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected["Gamma"]) * self._multiply_num

    def portfolio_vega(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected["Vega"]) * self._multiply_num

    def create_init_weight(self, hedge_size: float, p1: float = 0.5):
        """创建初始权重 (基于手动规则).

        Args:
            hedge_size: 对冲名义
            p1: 风险预算比例

        Returns:
            初始权重 Series
        """
        otm = self._selected.copy()
        weights = pd.Series(0.0, index=otm.index)

        # Level-1 put: hedge_size * p1
        p1_code = self.level_put(1)
        if p1_code:
            idx = otm[otm["contract_code"] == p1_code].index
            if len(idx) > 0:
                p1_delta = otm.loc[idx[0], "Delta"]
                weights.loc[idx[0]] = required_share(hedge_size * p1, p1_delta, self.f)

        # Level-1 call: hedge_size * p1 * 0.6
        c1_code = self.level_call(1)
        if c1_code:
            idx = otm[otm["contract_code"] == c1_code].index
            if len(idx) > 0:
                c1_delta = otm.loc[idx[0], "Delta"]
                weights.loc[idx[0]] = required_share(hedge_size * p1 * 0.6, c1_delta, self.f)

        return weights
