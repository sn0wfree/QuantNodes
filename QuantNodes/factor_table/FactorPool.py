# coding=utf-8
import hashlib
from collections import deque
from itertools import chain

import pandas as pd

from QuantNodes.factor_table.Factors import Factor, FactorCreator, FactorInfo


class SaveTools(object):
    @staticmethod
    def _hash_func(s, encoding='utf-8'):
        return hashlib.md5(str(s).encode(encoding)).hexdigest()

    @classmethod
    def _create_key_name(cls, name: str, cik_dts: (str, list), cik_iid: (str, list)):
        """

        :param name:
        :param cik_dts:
        :param cik_iid:
        :return:
        """
        cik_dts_hash = cls._hash_func(cik_dts)  # hashlib.md5(str(cik_dts).encode('utf-8')).hexdigest()
        cik_iid_hash = cls._hash_func(cik_iid)  # hashlib.md5(str(cik_iid).encode('utf-8')).hexdigest()
        key = name + "-" + cik_dts_hash + '-' + cik_iid_hash
        return key

    @staticmethod
    def _filter_fetched(fetched, cik_dts, cik_iid, cik=None):
        if cik is None:
            cik = ['cik_dts', 'cik_iid']
        dt_mask = ~fetched[cik[0]].isin(cik_dts)
        iid_mask = ~fetched[cik[1]].isin(cik_iid)
        return fetched[dt_mask & iid_mask]

    @staticmethod
    def _check_stored_var_consistent(old_f_ind, name):
        """

        :param old_f_ind:
        :param name:
        :return:
        """
        old_name = old_f_ind['var'].unique()  # should one string
        if len(old_name) > 1:  # if got 2 or more var,that means something wrong at this file
            raise ValueError('f_ind.var got two or more diff vars')
        old_name = old_name[0]  # get all variables which has joined with comma
        if name == old_name:  # check all variable should only store same var
            pass
        else:
            raise ValueError(f'stored var:{old_name} diff from {name}')


class FactorTools(SaveTools):

    @classmethod
    def save(cls, fetched, store_path, _cik_dts=None, _cik_ids=None, cik_cols=['cik_dts', 'cik_iid']):
        """
        store factor pool

        will create f_ind table to store


        :param store_path:
        :param _cik_dts:
        :param _cik_ids:
        :return:
        """

        var_cols = sorted(filter(lambda x: x not in cik_cols, fetched.columns.tolist()))  # sorted variables
        name = ','.join(var_cols)  # have join all variable with comma
        # load data
        with pd.HDFStore(store_path, mode="a", complevel=3, complib=None, fletcher32=False, ) as h5_conn:
            # check f_ind
            keys = h5_conn.keys()
            if '/f_ind' not in keys:  # check index table whether exists
                # create new one
                f_ind = fetched[cik_cols]
                cik_dts, cik_iid = sorted(f_ind['cik_dts'].values.tolist()), sorted(f_ind['cik_iid'].values.tolist())

                key = cls._create_key_name(name, cik_dts, cik_iid)
                f_ind['var'] = name  # string
                f_ind['store_loc'] = key
                h5_conn['f_ind'] = f_ind
                h5_conn[key] = fetched
            else:
                # update old one
                old_f_ind = h5_conn['f_ind']
                cls._check_stored_var_consistent(old_f_ind, name)  # name must be string
                cik_dts = old_f_ind['cik_dts'].values.tolist()
                cik_iid = old_f_ind['cik_iid'].values.tolist()
                new_fetched = cls._filter_fetched(fetched, cik_dts, cik_iid, cik=cik_cols)
                if new_fetched.empty:
                    print(f'new_fetched is empty! nothing to update!')
                else:
                    cik_dts = sorted(new_fetched['cik_dts'].values.tolist())
                    cik_iid = sorted(new_fetched['cik_iid'].values.tolist())
                    f_ind = fetched[cik_cols]
                    key = cls._create_key_name(name, cik_dts, cik_iid)
                    f_ind['store_loc'] = key
                    f_ind['var'] = name
                    new_f_ind = pd.concat([old_f_ind, f_ind])
                    h5_conn['f_ind'] = new_f_ind
                    h5_conn[key] = new_fetched


class FactorPool(deque):
    def __init__(self, *args, **kwargs):
        super(FactorPool, self).__init__(*args, **kwargs)
        self._cik_ids = None
        self._cik_dts = None

    @property
    def factors(self):
        h = []
        for i in self:
            ele = i.element
            if ele.as_alias is None:  # not set alias
                for f in ele.factor_names:
                    if f in h:
                        raise ValueError(f'factor got duplicated variable!{f}')
                    else:
                        h.append(f)
            else:
                for f in ele.as_alias:
                    if f in h:
                        raise ValueError(f'factor got duplicated variable!{f}')
                    else:
                        h.append(f)
        return h

    def set_cik_ids(self, cik_ids_list):
        self._cik_ids = list(cik_ids_list)

    def set_cik_dts(self, cik_dts_list):
        self._cik_dts = list(cik_dts_list)

    def add_factor(self, *args, **kwargs):
        if isinstance(args[0], Factor):
            factor = args[0]
        elif isinstance(args[0],Fac)
        else:
            factor = FactorCreator(*args, **kwargs)


        exist_factor = self.factors

        ele = factor.element
        if ele.as_alias is None:
            for f in ele.factor_names:
                if f in exist_factor:
                    raise ValueError(f'factor got duplicated variable!{f}')
        else:
            for f in ele.as_alias:
                if f in exist_factor:
                    raise ValueError(f'factor got duplicated variable!{f}')
        self.append(factor)

    @staticmethod
    def merge_df_factor(factor_list):
        res = list(filter(lambda x: x._obj_type == 'DF', factor_list))
        if len(res) == 0:
            yield None
        else:
            for df in res:
                yield df

    @staticmethod
    def merge_h5_factor(factor_list):
        res = list(filter(lambda x: x._obj_type == 'H5', factor_list))
        if len(res) == 0:
            yield None
        else:
            for h5 in res:
                yield h5

    @staticmethod
    def merge_sql_factor(factor_list):
        res = sorted(FactorInfo.__dataclass_fields__.keys())
        cols = list(res[1:4]) + [res[-1], res[4]]
        factor_name_col = res[5]
        alias_col = res[0]

        factors = list(filter(lambda x: x._obj_type.startswith(('SQL_', 'db_table')), factor_list))
        if len(factors) == 0:
            yield None
        else:
            factors_info = pd.DataFrame(list(map(lambda x: x.factor_info(), factors)))
            # dataframe have not reduced!
            for (db_table, dts, iid, via, info), df in factors_info.groupby(cols):
                # merge same source data
                cc = df[[factor_name_col, alias_col]].apply(lambda x: ','.join(x))
                origin_factor_names = cc[factor_name_col].split(',')
                alias = cc[alias_col].split(',')
                # use set will disrupt the order
                # we need keep the origin order
                back = list(zip(origin_factor_names, alias))
                disrupted = list(set(back))
                disrupted.sort(key=back.index)
                origin_factor_names_new, alias_new = zip(*disrupted)
                alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))
                idx_mask = df.index[0]
                f = factors[idx_mask]

                res = FactorCreator(db_table, f._obj, dts, iid, origin_factor_names_new, via,
                                    as_alias=alias_new,
                                    db_table=db_table, db_type=f.db_type)
                yield res

    def merge_factors(self):
        # todo unstack factor name to check whether factor exists duplicates!!
        factors = FactorPool()

        tasks = chain(self.merge_sql_factor(self), self.merge_df_factor(self), self.merge_h5_factor(self))

        for f in tasks:
            if f is not None:
                factors.append(f)
        # sql_factor = list(self.merge_sql_factor(self))
        # if sql_factor[0] is not None:
        #     factors.extend(sql_factor)
        # df_factor = list(self.merge_df_factor(self))
        # if df_factor[0] is not None:
        #     factors.extend(df_factor)
        # h5_factor = list(self.merge_h5_factor(self))
        # if h5_factor[0] is not None:
        #     factors.extend(h5_factor)
        return factors

    def check_cik_dts(self, _cik_dts):
        if _cik_dts is not None:
            self._cik_dts = _cik_dts
        else:
            if self._cik_dts is None:
                raise KeyError('cik_dts(either default approach or fetch) both are not setup!')

    def check_cik_iids(self, _cik_iids):
        if _cik_iids is not None:
            self._cik_ids = _cik_iids
        else:
            if self._cik_ids is None:
                raise KeyError('cik_ids(either default approach or fetch) both are not setup!')

    def save(self, store_path, _cik_dts=None, _cik_iids=None, reduced=True, cik_cols=['cik_dts', 'cik_iid']):
        """
        store factor pool

        will create f_ind table to store


        :param cik_cols:
        :param store_path:
        :param _cik_dts:
        :param _cik_iids:
        :param reduced:
        :return:
        """

        def filter_no_duplicated_fetched(fetched, old_f_ind, name, cik_cols=['cik_dts', 'cik_iid']):
            """

            :param fetched:
            :param old_f_ind:
            :param name:
            :param cik_cols:
            :return:
            """
            SaveTools._check_stored_var_consistent(old_f_ind, name)
            cik_dts = old_f_ind['cik_dts'].values.tolist()
            cik_iid = old_f_ind['cik_iid'].values.tolist()
            dt_mask = ~fetched[cik_cols[0]].isin(cik_dts)
            iid_mask = ~fetched[cik_cols[1]].isin(cik_iid)
            return fetched[dt_mask & iid_mask]

        fetched = self.fetch(_cik_dts=_cik_dts, _cik_ids=_cik_iids, reduced=reduced).reset_index()
        cols = sorted(filter(lambda x: x not in cik_cols, fetched.columns.tolist()))
        name = ','.join(cols)

        with pd.HDFStore(store_path, mode="a", complevel=3, complib=None, fletcher32=False, ) as h5_conn:
            # check f_ind
            keys = h5_conn.keys()
            if '/f_ind' not in keys:
                # create new one
                f_ind = fetched[cik_cols]
                cik_dts = sorted(f_ind['cik_dts'].values.tolist())  # record dts
                cik_iid = sorted(f_ind['cik_iid'].values.tolist())  # record iid
                key = SaveTools._create_key_name(name, cik_dts, cik_iid)
                f_ind['var'] = name
                f_ind['store_loc'] = key
                h5_conn['f_ind'] = f_ind
                h5_conn[key] = fetched
            else:
                # update old one
                old_f_ind = h5_conn['f_ind']
                new_fetched = filter_no_duplicated_fetched(fetched, old_f_ind, name, cik_cols=cik_cols)
                if new_fetched.empty:
                    print(f'new_fetched is empty! nothing to update!')
                else:
                    cik_dts = sorted(new_fetched['cik_dts'].values.tolist())
                    cik_iid = sorted(new_fetched['cik_iid'].values.tolist())
                    f_ind = fetched[cik_cols]
                    key = SaveTools._create_key_name(name, cik_dts, cik_iid)
                    f_ind['store_loc'] = key
                    f_ind['var'] = name
                    new_f_ind = pd.concat([old_f_ind, f_ind])
                    h5_conn['f_ind'] = new_f_ind
                    h5_conn[key] = new_fetched

    def load(self, store_path, cik_dts=None, cik_iids=None):

        with pd.HDFStore(store_path, mode="a", complevel=3, complib=None, fletcher32=False, ) as h5_conn:
            keys = h5_conn.keys()
            if '/f_ind' not in keys:
                raise ValueError(f'h5:{store_path} has no f_ind key! please check!')
            f_ind = h5_conn['f_ind']
            var = f_ind['var'].unique()
            if len(var) > 1: raise ValueError(f'f_ind.var got two or more diff vars:{var}')
            var = var[0].split(',')

            if cik_dts is None and cik_iids is None:
                store_loc_list = f_ind['store_loc'].unique()
                df = pd.concat([h5_conn[loc] for loc in store_loc_list])
                f1 = FactorCreator(store_path, df, 'cik_dts', 'cik_iid', factor_names=var)
                self.add_factor(f1)
            else:
                selected_cik_dts = f_ind['cik_dts'].unique() if cik_dts is None else cik_dts
                selected_cik_iid = f_ind['cik_iid'].unique() if cik_iids is None else cik_iids

                dts_mask = f_ind['cik_dts'].isin(selected_cik_dts)
                ids_mask = f_ind['cik_iid'].isin(selected_cik_iid)

                store_loc_list = f_ind[dts_mask & ids_mask]['store_loc'].unique()
                df = pd.concat([h5_conn[loc] for loc in store_loc_list])

                dts_mask = df['cik_dts'].isin(selected_cik_dts)
                ids_mask = df['cik_iid'].isin(selected_cik_iid)

                f1 = FactorCreator(store_path, df[dts_mask & ids_mask], 'cik_dts', 'cik_iid', factor_names=var)
                self.add_factor(f1)

    def fetch(self, _cik_dts=None, _cik_ids=None, reduced=True):
        """

        :param reduced: whether use reduce form
        :param _cik_dts: set up dts
        :param _cik_ids:  set up iids
        :return:
        """
        self.check_cik_iids(_cik_iids=_cik_ids)
        self.check_cik_dts(_cik_dts=_cik_dts)
        factors = self.merge_factors() if reduced else self
        fetched = (f.get(self._cik_dts, self._cik_ids).set_index(['cik_dts', 'cik_iid']) for f in factors)
        result = pd.concat(fetched, axis=1)
        # columns = result.columns.tolist()
        return result


if __name__ == '__main__':
    pass
