# coding=utf-8
import numpy as np
import pandas as pd
from QuantOPT.models import BaseModels
from itertools import permutations

# from QuantOPT.constraints.relaxer import RunOpt
# from QuantOPT.constraints.constraints import create_constraints_holder


import random
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


class RiskParity(BaseModels):
    @staticmethod
    def cal_port_var(w, cov):
        """

        :param w:
        :param cov:
        :return:
        """
        return np.sqrt(np.dot(np.dot(w, cov), w.T))

    @classmethod
    def mrc(cls, w, cov):
        """

        :param w:
        :param cov:
        :return:
        """
        port_std = cls.cal_port_var(w, cov)
        return np.dot(cov, w) / port_std

    @classmethod
    def rc(cls, w, cov):
        """

        :param w:
        :param cov:
        :return:
        """
        mrc = cls.mrc(w, cov)
        return pd.DataFrame([w_i * mrc_i for w_i, mrc_i in zip(w, mrc)]).values.ravel()

    @classmethod
    def loss_func(cls, w: np.array, **kwargs):
        """

        :param w: weight for min var
        :return: variance
        """

        cov = cls.get_cov(w)
        rc_list = cls.rc(w, cov)

        temp = np.nansum([np.power(a *100 - b*100, 2) for a, b in permutations(rc_list, 2)])
        return temp

    @classmethod
    def run_opt(cls, stockpool: (list, np.array), cov, bounds, constraints, method=None, **kwargs):
        """
        
        :param stockpool: the stockpool
        :param bounds: the bounds of weight
        :param constraints: the constraints of weight
        :param method: the method of optimization
        :param kwargs: the kwargs of optimization
        :return: the optimized weight
        """
        weight_length = len(stockpool)

        def get_cov(*args, **kwargs):
            return cov

        cls.get_cov = get_cov

        return cls.opt(bounds, constraints, weight_length, method=method, **kwargs)


class FactorRiskParity(BaseModels):
    @staticmethod
    def cal_port_var(w, B, cov):
        """

        :param w:
        :param B: factor exposure
        :param cov: factor cov
        :return:
        """
        W = np.dot(w, B)
        return np.sqrt(np.dot(np.dot(W, cov), W.T))

    @classmethod
    def mrc(cls, w, B, cov):
        """

        :param w:
        :param B:
        :param cov:
        :return:
        """
        W = np.dot(w, B)

        port_std = cls.cal_port_var(w, B, cov)

        return np.dot(cov, W) / port_std

    @classmethod
    def rc(cls, w, B, cov):
        """

        :param w:
        :param B:
        :param cov:
        :return:
        """
        W = np.dot(w, B)
        mrc = cls.mrc(w, B, cov)
        return pd.DataFrame([w_i * mrc_i for w_i, mrc_i in zip(W, mrc)]).values.ravel()

    @classmethod
    def loss_func(cls, w: np.array, **kwargs):
        """

        :param w: weight for min var
        :return: variance
        """

        cov = cls.get_factor_cov(w)
        B = cls.get_factor_expo(w)
        rc_list = cls.rc(w, B, cov)

        temp = np.nansum([np.power(a* 100 - b*100 , 2) for a, b in permutations(rc_list, 2)])
        return temp

    @classmethod
    def run_opt(cls, stockpool: (list, np.array), factor_cov, factor_expo, bounds, constraints, method=None, **kwargs):
        """

        :param stockpool: the stockpool
        :param factor_cov: factor_cov
        :param factor_expo: factor_exposure
        :param bounds: the bounds of weight
        :param constraints: the constraints of weight
        :param method: the method of optimization
        :param kwargs: the kwargs of optimization
        :return: the optimized weight
        """
        weight_length = len(stockpool)

        def get_factor_cov(*args, **kwargs):
            return factor_cov

        def get_factor_expo(*args, **kwargs):
            return factor_expo

        cls.get_factor_cov = get_factor_cov
        cls.get_factor_expo = get_factor_expo

        return cls.opt(bounds, constraints, weight_length, method=method, **kwargs)


if __name__ == '__main__':
    from QuantOPT.core.model_core import Holder

    Holder.add_model('RiskParity', RiskParity)
    Holder.add_model('FactorRiskParity', FactorRiskParity)
    pass
