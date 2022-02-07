# coding=utf-8
import unittest
import pandas as pd
import numpy as np

from QuantNodes.factor_table.Factors import FactorCreator, Factor


class MyTestCase(unittest.TestCase):
    def test_class_type_df(self):
        df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
        f1 = FactorCreator('test', df, 'cik_dts', 'cik_iid', factor_names=['v1', 'v2'])
        self.assertIsInstance(f1, Factor)
        r1 = f1.get(df['cik_dts'].values.tolist(), df['cik_iid'].values.tolist())
        self.assertListEqual(df.values.ravel().tolist(), r1.values.ravel().tolist())

    def test_class_type_h5(self):
        df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
        df.to_hdf('test.h5', 'test')
        test_h5 = 'test.h5'
        f1 = FactorCreator('test', test_h5, 'cik_dts', 'cik_iid', factor_names=['v1', 'v2'])
        self.assertIsInstance(f1, Factor)
        r1 = f1.get(df['cik_dts'].values.tolist(), df['cik_iid'].values.tolist())
        self.assertListEqual(df.values.ravel().tolist(), r1.values.ravel().tolist())


if __name__ == '__main__':
    unittest.main()
