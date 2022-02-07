# coding=utf-8
import unittest
import numpy as np
import pandas as pd
from QuantNodes.factor_table.FactorPool import FactorPool, FactorCreator


class MyTestCase(unittest.TestCase):
    def test_alias_df(self):
        np.random.seed(1012)
        # from ClickSQL import BaseSingleFactorTableNode
        #
        # src = 'clickhouse://default:***REMOVED***@***REMOVED***:8123/system'
        # node = BaseSingleFactorTableNode(src)
        # f2_ = FactorPool()
        #
        # df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
        # f1 = FactorCreator('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', factor_names=['zone', 'cik'],
        #                    db_type='ClickHouse')
        # f2_.add_factor(f1)
        # f2_.add_factor('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', factor_names='size,crawler', db_type='ClickHouse')
        # ip = ['107.23.85.jfd', '104.198.37.bdc', '104.155.169.jbh']
        # f_ip = "','".join(ip)
        # sql = f"select date,ip,idx,norefer,noagent from EDGAR_LOG.parsed_edgar_log where ip in ('{f_ip}')  "
        # df = node(sql)
        # f2_.add_factor('parsed_edgar_log', df, 'date', 'ip', factor_names='idx,norefer,noagent')

        #
        # f3 = FactorUnit('test.test2', query, 'cik_dts', 'cik_iid', factor_names=['v3'])
        #

        # f2_.add_factor('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', 'code')
        # f2_.add_factor('select * from EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', 'idx,cik,noagent')
        # f2_.merge_factors(inplace=True)
        # df = f2_.fetch(_cik_dts=['2017-04-30'], _cik_iids=ip)
        f2_ = FactorPool()
        # df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 3)), columns=['cik_dts', 'cik_iid', 'v1', ])
        # f1 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v1'])
        # f2_.add_factor(f1)
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 3)), columns=['cik_dts', 'cik_iid', 'v22'])
        f11 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v22'],
                            as_alias=['test_v22'])
        f2_.add_factor(f11)
        # cik_iid = list(set(f1.get_cik_iid() + f11.get_cik_iid()))
        # cik_dts = list(set(f1.get_cik_dts() + f11.get_cik_dts()))
        # f2_.set_cik_dts(cik_dts)
        # f2_.set_cik_iids(cik_iid)

        # f2_.load('test.h5')
        self.assertListEqual(f2_.factors, ['test_v22'])
        # print()
        # df = f2_.fetch(_cik_dts=cik_dts, _cik_iids=cik_iid)
        self.assertEqual(True, True)  # add assertion here

    def test_2_factor_pool(self):
        np.random.seed(1012)
        f2_ = FactorPool()
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 3)), columns=['cik_dts', 'cik_iid', 'v1', ])
        f1 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v1'])
        f2_.add_factor(f1)
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 3)), columns=['cik_dts', 'cik_iid', 'v22'])
        f11 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v22'],
                            as_alias=['test_v22'])
        f2_.add_factor(f11)
        self.assertListEqual(f2_.factors, ['v1', 'test_v22'])

    def test_2_factor_pool_2(self):
        np.random.seed(1012)
        f2_ = FactorPool()
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
        f1 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v1'])
        f2_.add_factor(f1)
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v11', 'v22'])
        f11 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v11', 'v22'],
                            as_alias=[None, 'test_v22'])
        f2_.add_factor(f11)
        self.assertListEqual(f2_.factors, ['v1', 'v11', 'test_v22'])

    def test_2_factor_pool_3(self):
        np.random.seed(1012)
        f2_ = FactorPool()
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 5)),
                          columns=['cik_dts', 'cik_iid', 'v1', 'v2', 'v3'])
        f1 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v1', 'v2'])
        f2_.append(f1)
        df = pd.DataFrame(np.random.randint(0, 100000, size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v11', 'v22'])
        f11 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v11', 'v22'],
                            as_alias=[None, 'test_v22'])
        f2_.add_factor(f11)
        self.assertListEqual(f2_.factors, ['v1', 'v2', 'v11', 'test_v22'])


if __name__ == '__main__':
    unittest.main()
