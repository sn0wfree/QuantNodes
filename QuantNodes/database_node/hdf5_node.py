# coding=utf-8
from functools import lru_cache

import pandas as pd


class HDFNode(object):
    """

    save data
    load data
    update data
    """
    @staticmethod
    def _save_data(data: pd.DataFrame, key: str, data_path: str):
        """
        save data into hdf5
        :param data:
        :param key:
        :param data_path:
        :return:
        """
        with pd.HDFStore(data_path, mode='r') as f:
            keys = list(map(lambda x: x[1:], f.keys()))
            if key in keys:
                raise ValueError(f'found duplicated key:{key}, please use UpdateData to save data')
            else:
                f[key] = data

    @staticmethod
    @lru_cache(max=200)
    def _rm_slash_from_keys(keys_with_slash):
        keys = list(map(lambda x: x[1:], keys_with_slash))
        return keys

    @classmethod
    def _check_keys(cls, key, keys_with_slash):
        keys = cls._rm_slash_from_keys(keys_with_slash)
        return key in keys

    @classmethod
    def _update_data(cls, data: pd.DataFrame, key: str, data_path: str, ):
        """

        存成 stk，dt, data1,data2,data3 结构


        :param data:
        :param key:
        :param data_path:
        :return:
        """
        with pd.HDFStore(data_path, mode='r') as f:

            if cls._check_keys(key, f.keys()):
                f[key] = data
            else:
                old_data = f.get(key)
                if sorted(data.columns) != sorted(old_data.columns):
                    raise ValueError(
                        f'updated data own different column from old data!, please check key or data source! key param: {key}')
                else:
                    f[key] = pd.concat([data, old_data])

    # @classmethod
    # def add_column(cls,data: pd.DataFrame, key: str, data_path: str, stk, dt):
    #     with pd.HDFStore(data_path, mode='r') as f:
    #         if not cls._check_keys(key, f.keys()):
    #             raise KeyError(f'{key} not exists!')
    #         else:
    #
    #     pass

    @classmethod
    def _get_stored_keys(cls, data_path: str):
        """

        :param data_path:
        :return:
        """
        with pd.HDFStore(data_path, mode='r') as f:
            return cls._rm_slash_from_keys(f.keys())

    @classmethod
    def _get_stored_data(cls, key: str, data_path: str):
        """
        get data
        :param key:
        :param data_path:
        :return:
        """
        with pd.HDFStore(data_path, mode='r') as f:
            if cls._check_keys(key, f.keys()):
                return f.get(key)
            else:
                raise ValueError(f'{data_path}[{key}]数据不存在, only found {",".join(f.keys())}')

    @classmethod
    def _get_stored_type_asset_list(cls, apidata_dir: str, dt_col: str, id_col: str):
        keys = cls._get_stored_keys(apidata_dir)
        if dt_col in keys:
            trade_dt = cls._get_stored_data(dt_col, apidata_dir)
        else:
            raise ValueError(f'{dt_col} column is not exists')
        if id_col in keys:
            asset_list = cls._get_stored_data(id_col, apidata_dir)
        else:
            raise ValueError(f'{id_col} column is not exists')
        return asset_list, trade_dt
        pass


if __name__ == '__main__':
    pass
