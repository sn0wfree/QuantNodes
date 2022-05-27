# coding=utf-8

import pandas as pd
import os

path = 'C:\\Users\\user\\Documents\\GitHub\\QuantNodes\\data\\test_h5_new'


class FakeDataGet(object):
    """
     ep = pd.read_hdf(os.path.join(path, 'alpha\\ep.h5'))
    bp = pd.read_hdf(os.path.join(path, 'alpha\\bp.h5'))

    ep_ind_index = pd.read_hdf(os.path.join(path, 'alpha\\ep_ind_index.h5'))
    ep_ind_index_per = pd.read_hdf(os.path.join(path, 'alpha\\ep_ind_index_per.h5'))
    ep_ind_index_per_1Y = pd.read_hdf(os.path.join(path, 'alpha\\ep_ind_index_per_1Y.h5'))
    k_spm_new = pd.read_hdf(os.path.join(path, 'alpha\\k_spm_new.h5'))

    mom_21day = pd.read_hdf(os.path.join(path, 'alpha\\mom_21day.h5'))
    """

    def alpha_get(self, key):
        ep = pd.read_hdf(os.path.join(path, f'alpha\\{key}.h5'))
        return ep

    @staticmethod
    def load_index_daily():
        with pd.HDFStore(os.path.join(path, 'index_daily.h5')) as f:
            trade_dt = f['trade_dt'].values.ravel()
            indexlist = f['indexlist'].values.ravel()
            idx_name_df = pd.concat([f['indexlist'], f['indexname']], axis=1)
            data = pd.DataFrame(f['index_cp'])
            data.columns = indexlist
            data.index = trade_dt
        return data, idx_name_df


class STKDaily(object):
    def __init__(self, name='stk_daily.h5'):
        self.full_path = os.path.join(path, name)
        with pd.HDFStore(self.full_path) as f:
            self.keys = list(map(lambda x: x[1:], f.keys()))
            # print(self.keys)
            self.trade_dt = f['trade_dt'].values.ravel()
            self.stklist = f['stklist'].values.ravel()

    def get(self, key):
        if key in self.keys:
            with pd.HDFStore(self.full_path) as f:
                temp = pd.DataFrame(f[key])
                temp.columns = self.stklist
                temp.index = self.trade_dt
                return temp
        else:
            raise ValueError('key not exists!')

    def get_stk_name_df(self):
        with pd.HDFStore(self.full_path) as f:
            stk_name_df = pd.concat([f['stklist'], f['stkname']], axis=1)
            stk_name_df.columns = ['stklist', 'stkname']
        return stk_name_df


class RiskFactor(object):
    def __init__(self):
        with pd.HDFStore(os.path.join(path, 'stk_daily.h5')) as f:
            self.trade_dt = f['trade_dt'].values.ravel()
            self.stklist = f['stklist'].values.ravel()
        with pd.HDFStore(os.path.join(path, 'risk_factor.h5')) as f2:
            self.keys = list(map(lambda x: x[1:], f2.keys()))

    def get(self, key):
        if key in self.keys:
            with pd.HDFStore(os.path.join(path, 'risk_factor.h5')) as f2:
                temp = f2[key]
                temp.columns = self.stklist
                temp.index = self.trade_dt
                return temp
        else:
            raise ValueError(f'{key} is not exists!')





if __name__ == '__main__':
    # {'name': 'ep', 'factor_dir': './testdata/test_h5_new/alpha/ep.h5'}
    # stk_daily = STKDaily()
    ['/amt', '/cp', '/hp', '/id_300', '/id_50', '/id_500', '/id_citic1', '/id_citic1A', '/ind_code_CITIC_1',
     '/ind_code_CITIC_1A', '/ind_name_CITIC_1', '/ind_name_CITIC_1A', '/ipo_date', '/ipo_days', '/lp', '/mv_float',
     '/op', '/st', '/stklist', '/stkname', '/suspend', '/tmv', '/trade_dt', '/ud_limit', '/vol', '/weight_300',
     '/weight_50', '/weight_500']

    RF = RiskFactor()
    print(RF.keys)
    # Beta = RF.get('Beta')

    # stk_name_df = stk_daily.get_stk_name_df()
    # amt = stk_daily.get('amt')
    # cp = stk_daily.get('cp')
    # print(stk_daily.keys)
    # cp = pd.DataFrame(f['cp'], columns=stklist, index=trade_dt)
    # trade_dt = f['trade_dt'].values.ravel()
    # indexlist = f['indexlist'].values.ravel()
    # idx_name_df = pd.concat([f['indexlist'], f['indexname']])
    # data = pd.DataFrame(f['index_cp'], columns=indexlist, index=trade_dt)

    print(1)
    pass
