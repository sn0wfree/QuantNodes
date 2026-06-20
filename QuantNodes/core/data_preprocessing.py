# coding=utf-8
"""
数据预处理工具类

替代 QuantStudio.FactorDataBase.FactorOperation.DataPreprocessingFun
提供因子数据预处理的各种操作：标准化、去极值、缺失值处理、正交化等
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Callable


class DataPreprocessingFun:
    """
    数据预处理函数集合

    提供因子数据预处理的各种静态方法，包括：
    - 标准化：Z-Score、Rank、分位数
    - 去极值：Winsorize
    - 缺失值填充：常量、函数、回归
    - 正交化：Gram-Schmidt
    """

    @staticmethod
    def regressChangeRate(x: np.ndarray) -> np.ndarray:
        """
        回归变化率

        对输入数据进行回归分析，返回回归系数作为变化率度量
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim < 2:
            return x

        n_factor = x.shape[0] if x.ndim > 2 else 1
        result = np.zeros(x.shape[1:]) if x.ndim > 2 else np.zeros(x.shape)

        if x.ndim > 2:
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    y = x[0, i, j] if x.shape[0] > 0 else x[i, j]
                    if n_factor > 1:
                        X = np.vstack([x[k, i, j] for k in range(1, n_factor)]).T
                        X = np.column_stack([np.ones(len(X)), X])
                        try:
                            beta = np.linalg.lstsq(X, y, rcond=None)[0]
                            result[i, j] = beta[1] if len(beta) > 1 else beta[0]
                        except Exception:
                            result[i, j] = np.nan
                    else:
                        result[i, j] = y[-1] - y[0] if len(y) > 1 else y[0]
        else:
            for i in range(x.shape[1]):
                y = x[:, i]
                t = np.arange(len(y))
                try:
                    beta = np.polyfit(t, y, 1)[0]
                    result[i] = beta
                except Exception:
                    result[i] = np.nan

        return result

    @staticmethod
    def standardizeZScore(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        avg_weight: Optional[np.ndarray] = None,
        dispersion_weight: Optional[np.ndarray] = None,
        avg_statistics: str = "平均值",
        dispersion_statistics: str = "标准差",
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """Z-Score 标准化"""
        result = data.copy().astype(float)
        valid_mask = ~np.isnan(data)

        if mask is not None:
            valid_mask = valid_mask & (mask == 1)

        if cat_data is None:
            if avg_weight is not None and dispersion_weight is not None:
                mean_val = np.nansum(data * avg_weight) / np.nansum(avg_weight * valid_mask)
                weighted_diff_sq = ((data - mean_val) * avg_weight) ** 2
                std_val = np.sqrt(
                    np.nansum(weighted_diff_sq) / np.nansum(avg_weight * valid_mask)
                )
            else:
                mean_val = np.nanmean(data[valid_mask]) if valid_mask.any() else 0
                std_val = np.nanstd(data[valid_mask]) if valid_mask.any() else 1

            if std_val == 0 or np.isnan(std_val):
                std_val = 1

            result[valid_mask] = (data[valid_mask] - mean_val) / std_val
            result[~valid_mask] = np.nan
        else:
            unique_cats = np.unique(cat_data[~np.isnan(cat_data)])
            for cat in unique_cats:
                cat_mask = valid_mask & (cat_data == cat)
                if cat_mask.any():
                    cat_data_arr = data[cat_mask]
                    if avg_weight is not None:
                        w = avg_weight[cat_mask]
                        cat_weight = w * ~np.isnan(cat_data_arr)
                        mean_val = np.nansum(cat_data_arr * w) / np.nansum(cat_weight)
                        cat_diff_sq = ((cat_data_arr - mean_val) * w) ** 2
                        std_val = np.sqrt(np.nansum(cat_diff_sq) / np.nansum(cat_weight))
                    else:
                        mean_val = np.nanmean(cat_data_arr)
                        std_val = np.nanstd(cat_data_arr)

                    if std_val == 0 or np.isnan(std_val):
                        std_val = 1

                    result[cat_mask] = (data[cat_mask] - mean_val) / std_val

            uncategorized_mask = valid_mask & np.isnan(cat_data)
            if uncategorized_mask.any():
                result[uncategorized_mask] = np.nan

        return result

    @staticmethod
    def standardizeRank(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        ascending: bool = True,
        uniformization: bool = True,
        perturbation: bool = False,
        offset: float = 0.5,
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """Rank 标准化"""
        result = np.zeros_like(data, dtype=float)
        valid_mask = ~np.isnan(data)

        if mask is not None:
            valid_mask = valid_mask & (mask == 1)

        if cat_data is None:
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                ranks = stats.rankdata(valid_data, ascending=ascending)
                if uniformization:
                    ranks = ranks / (len(ranks) + 1)
                if perturbation:
                    ranks = ranks + np.random.uniform(-offset, offset, len(ranks))
                result[valid_mask] = ranks
            result[~valid_mask] = np.nan
        else:
            unique_cats = np.unique(cat_data[~np.isnan(cat_data)])
            for cat in unique_cats:
                cat_mask = valid_mask & (cat_data == cat)
                valid_cat_data = data[cat_mask]
                if len(valid_cat_data) > 0:
                    ranks = stats.rankdata(valid_cat_data, ascending=ascending)
                    if uniformization:
                        ranks = ranks / (len(ranks) + 1)
                    if perturbation:
                        ranks = ranks + np.random.uniform(-offset, offset, len(ranks))
                    result[cat_mask] = ranks

            uncategorized_mask = valid_mask & np.isnan(cat_data)
            if uncategorized_mask.any():
                result[uncategorized_mask] = np.nan

        return result

    @staticmethod
    def fillNaNByValue(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        fill_value: float = 0,
        fill_method: str = "常数",
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """按值填充 NaN"""
        result = data.copy()
        nan_mask = np.isnan(data)

        if mask is not None:
            nan_mask = nan_mask & (mask == 1)

        if fill_method == "常数":
            result[nan_mask] = fill_value
        elif fill_method == "均值":
            valid_data = data[~nan_mask]
            fill_val = np.nanmean(valid_data) if len(valid_data) > 0 else fill_value
            result[nan_mask] = fill_val
        elif fill_method == "中位数":
            valid_data = data[~nan_mask]
            fill_val = np.nanmedian(valid_data) if len(valid_data) > 0 else fill_value
            result[nan_mask] = fill_val
        elif fill_method == "前向填充":
            df = pd.DataFrame(result)
            result = df.ffill().values
            result[nan_mask] = np.nan
        elif fill_method == "后向填充":
            df = pd.DataFrame(result)
            result = df.bfill().values
            result[nan_mask] = np.nan

        return result

    @staticmethod
    def winsorize(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        winsorize_lower: float = 0.01,
        winsorize_upper: float = 0.01,
        fill_value: Optional[float] = None,
        fill_method: str = "均值方差",
        boundary_method: str = "边界值",
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """Winsorize 去极值处理"""
        if not (0 <= winsorize_lower <= 1) or not (0 <= winsorize_upper <= 1):
            raise ValueError(
                f"winsorize bounds must be in [0, 1], "
                f"got lower={winsorize_lower}, upper={winsorize_upper}"
            )
        if winsorize_lower + winsorize_upper > 1:
            total = winsorize_lower + winsorize_upper
            raise ValueError(
                f"winsorize_lower + winsorize_upper must be <= 1, got {total}"
            )

        result = data.copy().astype(float)
        valid_mask = ~np.isnan(data)

        if mask is not None:
            valid_mask = valid_mask & (mask == 1)

        if cat_data is None:
            valid_data = data[valid_mask]
            if len(valid_data) > 0:
                lower_bound = np.nanpercentile(valid_data, winsorize_lower * 100)
                upper_bound = np.nanpercentile(valid_data, (1 - winsorize_upper) * 100)

                if fill_method == "均值方差":
                    mean_val = np.nanmean(valid_data)
                    std_val = np.nanstd(valid_data)
                    result[valid_mask & (data < lower_bound)] = mean_val - 2 * std_val
                    result[valid_mask & (data > upper_bound)] = mean_val + 2 * std_val
                else:
                    result[valid_mask & (data < lower_bound)] = lower_bound
                    result[valid_mask & (data > upper_bound)] = upper_bound
        else:
            unique_cats = np.unique(cat_data[~np.isnan(cat_data)])
            for cat in unique_cats:
                cat_mask = valid_mask & (cat_data == cat)
                valid_cat_data = data[cat_mask]
                if len(valid_cat_data) > 0:
                    lower_bound = np.nanpercentile(valid_cat_data, winsorize_lower * 100)
                    upper_bound = np.nanpercentile(valid_cat_data, (1 - winsorize_upper) * 100)

                    if fill_method == "均值方差":
                        mean_val = np.nanmean(valid_cat_data)
                        std_val = np.nanstd(valid_cat_data)
                        result[cat_mask & (data < lower_bound)] = mean_val - 2 * std_val
                        result[cat_mask & (data > upper_bound)] = mean_val + 2 * std_val
                    else:
                        result[cat_mask & (data < lower_bound)] = lower_bound
                        result[cat_mask & (data > upper_bound)] = upper_bound

        return result

    @staticmethod
    def standardizeQuantile(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        ascending: bool = True,
        perturbation: bool = False,
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """分位数标准化"""
        return DataPreprocessingFun.standardizeRank(
            data=data,
            mask=mask,
            cat_data=cat_data,
            ascending=ascending,
            uniformization=True,
            perturbation=perturbation,
            offset=0.5,
            other_handle=other_handle,
            **kwargs
        )

    @staticmethod
    def fillNaNByFun(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        val_fun: Optional[Callable] = None,
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """按函数值填充 NaN"""
        result = data.copy().astype(float)
        nan_mask = np.isnan(data)

        if mask is not None:
            nan_mask = nan_mask & (mask == 1)

        if val_fun is None:
            val_fun = lambda x, n: np.zeros(n) + np.nanmean(x)

        n = np.sum(nan_mask)
        if n > 0:
            non_nan_data = data[~nan_mask]
            if len(non_nan_data) > 0:
                fill_values = val_fun(non_nan_data, n)
            else:
                fill_values = np.zeros(n) + np.nan
            result[nan_mask] = fill_values[:n]

        return result

    @staticmethod
    def fillNaNByRegress(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        intercept: bool = True,
        weight_data: Optional[np.ndarray] = None,
        dummy_data: Optional[np.ndarray] = None,
        other_handle: str = "填充None",
        **kwargs
    ) -> np.ndarray:
        """按回归预测值填充 NaN"""
        result = data.copy().astype(float)
        nan_mask = np.isnan(data)

        if mask is not None:
            nan_mask = nan_mask & (mask == 1)

        if X is None or X.size == 0:
            return result

        valid_rows = ~nan_mask
        nan_rows = nan_mask

        if np.sum(valid_rows) < 2 or np.sum(nan_rows) < 1:
            return result

        X_valid = X[valid_rows] if X.ndim > 1 else X[valid_rows].reshape(-1, 1)
        y_valid = data[valid_rows]

        if intercept:
            X_valid = np.column_stack([np.ones(len(X_valid)), X_valid])

        try:
            beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]

            X_nan = X[nan_rows] if X.ndim > 1 else X[nan_rows].reshape(-1, 1)
            if intercept:
                X_nan = np.column_stack([np.ones(len(X_nan)), X_nan])

            y_pred = X_nan @ beta
            result[nan_rows] = y_pred
        except Exception:
            pass

        return result

    @staticmethod
    def orthogonalize(
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        cat_data: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        method: str = "gram_schmidt",
        weight_data: Optional[np.ndarray] = None,
        dummy_data: Optional[np.ndarray] = None,
        other_handle: str = "填充None",
        intercept: bool = True,
        **kwargs
    ) -> np.ndarray:
        """正交化处理"""
        result = data.copy().astype(float)
        valid_mask = ~np.isnan(data)

        if mask is not None:
            valid_mask = valid_mask & (mask == 1)

        if X is None or X.size == 0:
            return result

        if method == "gram_schmidt":
            for i in range(result.shape[0]):
                row_mask = valid_mask[i]
                if not row_mask.any():
                    continue

                x_valid = X[row_mask]
                y_valid = result[i, row_mask]

                if len(x_valid) < 2:
                    continue

                try:
                    x_mean = np.nanmean(x_valid, axis=0)
                    x_centered = x_valid - x_mean
                    norm = np.linalg.norm(x_centered, axis=0)
                    norm[norm == 0] = 1
                    x_orthogonal = x_centered / norm

                    beta = np.linalg.lstsq(x_orthogonal, y_valid, rcond=None)[0]
                    y_pred = x_orthogonal @ beta
                    result[i, row_mask] = y_valid - y_pred + np.nanmean(y_valid)
                except Exception:
                    continue
        else:
            for i in range(result.shape[0]):
                row_mask = valid_mask[i]
                if not row_mask.any():
                    continue

                x_valid = X[row_mask]
                y_valid = result[i, row_mask]

                if len(x_valid) < 2:
                    continue

                try:
                    if intercept:
                        X_design = np.column_stack([np.ones(len(x_valid)), x_valid])
                    else:
                        X_design = x_valid

                    beta = np.linalg.lstsq(X_design, y_valid, rcond=None)[0]
                    y_pred = X_design @ beta
                    result[i, row_mask] = y_valid - y_pred + np.nanmean(y_valid)
                except Exception:
                    continue

        return result
