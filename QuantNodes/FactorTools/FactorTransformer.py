# coding=utf-8
import numpy as np
import pandas as pd
import statsmodels.api as sm


# tools for factor transform
## 中性化
class FactorNeutralization(object):
    """
    do factor Neutraliztion process
    """

    @staticmethod
    def neutralize(factor_df: pd.DataFrame, y_list: (list, str), ind_col: str = None, mktcap_col: str = None,
                   other_numeric_cols: list = None, add_const: bool = False):
        """
        do factor neutralization
        :return:
        """

        neutralize_cols = []
        if ind_col is not None:
            ind_list = factor_df[ind_col].unique().to_list()
            class_ind = pd.get_dummies(factor_df[ind_col], prefix=['industry'], prefix_sep="_",
                                       dummy_na=False, drop_first=False)

            neutralize_cols.extend(class_ind.columns.tolist())
        if mktcap_col is not None:
            neutralize_cols.append(mktcap_col)
        if other_numeric_cols is not None:
            neutralize_cols.extend(other_numeric_cols)
        if len(neutralize_cols) == 0:
            raise ValueError("no neutralize cols")

        # get factor_df
        if isinstance(y_list, str):
            y_list = [y_list]
        elif isinstance(y_list, (list, tuple)):
            pass
        else:
            raise ValueError("y_list type error, only aceept str or list, but got {}".format(type(y_list)))
        data = pd.cocnat([factor_df, class_ind])
        for y in y_list:
            if add_const:
                formula = f"{y} ~ 1 + {' + '.join(neutralize_cols)}"
            else:
                formula = f"{y} ~ {' + '.join(neutralize_cols)}"
            model = sm.OLS.from_formula(formula, data=data).fit()

            factor_df[y + "_neutralized"] = model.resid
        return factor_df

    pass


## 正交化
class FactorOrthogonalization(object):
    """
    施密特正交化(schmidt orthogonalization)
    规范正交化(canonial orthogonalization)
    对称正交化(symmetric orthogonalization)

    """

    @staticmethod
    def schmidt_orthogonalization(factor_df: pd.DataFrame, target_cols: list):
        martix = factor_df[target_cols].copy()

        R = np.zeros(martix.shape[1], martix.shape[1])

        Q = np.zeros(martix.shape)
        for k in range(martix.shape[1]):
            R[k, k] = np.sqrt(np.sum(martix[:, k] ** 2))
            Q[:, k] = martix[:, k] / R[k, k]
            for j in range(k + 1, martix.shape[1]):
                R[k, j] = np.dot(Q[:, k], martix[:, j])
                martix[:, j] = martix[:, j] - R[k, j] * Q[:, k]
        factor_df[[f"schmidt_orthed_{s}" for s in target_cols]] = Q
        return factor_df

    def canonial_orthogonalization(self, factor_df: pd.DataFrame, target_cols: list):
        martix = factor_df[target_cols].copy()
        Q = np.zeros(martix.shape)
        D, U = np.linalg.eig(martix.T @ martix)
        S = np.dot(U, np.diag(np.sqrt(D)))
        Q = np.dot(martix, S)
        factor_df[[f"canonial_orthed_{s}" for s in target_cols]] = Q
        return factor_df

    def symmetric_orthogonalization(self, factor_df: pd.DataFrame, target_cols: list):
        martix = factor_df[target_cols].copy()
        Q = np.zeros(martix.shape)
        D, U = np.linalg.eig(martix.T @ martix)
        S = np.dot(U, np.diag(np.sqrt(D)))
        Q = np.dot(martix, S)
        Q = np.dot(Q, U.T)
        factor_df[[f"symmetric_orthed_{s}" for s in target_cols]] = Q
        return factor_df

        pass


## 去极值
class FactorRemoveOutlier(object):
    """
    do factor remove outlier process
    """

    @staticmethod
    def mean_variance_outlier_clip(factor_Series: (pd.Series, np.array), threshold: int = 3):
        """
        remove outlier by mean variance
        :param factor_Series:
        :return:
        """
        factor_Series_outlier_removed = factor_Series.copy()
        avg = factor_Series_outlier_removed.mean()
        std = factor_Series_outlier_removed.std()
        factor_Series_outlier_removed = np.clip(factor_Series_outlier_removed, avg - threshold * std,
                                                avg + threshold * std)
        return factor_Series_outlier_removed

    @staticmethod
    def mad_outlier_clip(factor_Series: (pd.Series, np.array), threshold: int = 3, k=1.4826):
        """
        remove outlier by mad
        :param factor_Series:
        :return:
        """
        factor_Series_outlier_removed = factor_Series.copy()
        median = factor_Series_outlier_removed.median()
        mad = np.abs(factor_Series_outlier_removed - median).median()
        factor_Series_outlier_removed = np.clip(factor_Series_outlier_removed, median - threshold * k * mad,
                                                median + threshold * k * mad)
        return factor_Series_outlier_removed

    @staticmethod
    def winsorize_outlier_clip(factor_Series: (pd.Series, np.array), up_threshold: float = 0.95,
                               down_threshold: float = 0.05):
        """
        remove outlier by winsorize
        :param factor_Series:
        :return:
        """
        factor_Series_outlier_removed = factor_Series.copy()
        factor_Series_outlier_removed = factor_Series_outlier_removed.clip(
            lower=factor_Series_outlier_removed.quantile(down_threshold),
            upper=factor_Series_outlier_removed.quantile(up_threshold))
        return factor_Series_outlier_removed

    pass


## 归一化
class FactorNormalization(object):
    @staticmethod
    def min_max_normalization(factor_Series: (pd.Series, np.array)):
        """
        do factor min-max standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = (factor_Series_standardized - factor_Series_standardized.min()) / (
                factor_Series_standardized.max() - factor_Series_standardized.min())
        return factor_Series_standardized

    @staticmethod
    def mean_normalization(factor_Series: (pd.Series, np.array)):
        """
        do factor mean standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = (factor_Series_standardized - factor_Series_standardized.mean()) / (
                factor_Series_standardized.max() - factor_Series_standardized.min())
        return factor_Series_standardized

    @staticmethod
    def log_normalization(factor_Series: (pd.Series, np.array)):
        """
        do factor log standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = np.log(factor_Series_standardized)
        return factor_Series_standardized

    @staticmethod
    def softmoid(factor_Series: (pd.Series, np.array), a=1, b=0):
        """
        do factor softmoid standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = 1 / (1 + np.exp(-a * (factor_Series_standardized - b)))
        return factor_Series_standardized

    @staticmethod
    def softmax(factor_Series: (pd.Series, np.array)):
        """
        do factor softmax standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = np.exp(factor_Series_standardized) / np.exp(factor_Series_standardized).sum()
        return factor_Series_standardized


## 标准化
class FactorStandardization(object):
    def z_score_standardization(self, factor_Series: (pd.Series, np.array)):
        """
        do factor t-stat standardization
        :param factor_Series:
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = factor_Series_standardized / factor_Series_standardized.std()
        return factor_Series_standardized

    def central_standardization(self, factor_Series: (pd.Series, np.array)):
        """
        do factor central standardization
        :param factor_Series:1
        :return:
        """
        factor_Series_standardized = factor_Series.copy()
        factor_Series_standardized = factor_Series_standardized - factor_Series_standardized.mean()
        return factor_Series_standardized


if __name__ == '__main__':
    pass
