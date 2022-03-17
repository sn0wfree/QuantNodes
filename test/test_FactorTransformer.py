import unittest

import numpy as np
from QuantNodes.FactorTools.FactorTransformer import FactorRemoveOutlier


class MyTestCaseFactorRemoveOutlier(unittest.TestCase):
    def test_mean_variance_outlier_clip(self):
        np.random.seed(1)
        data = (np.random.random(size=(100,)) - 0.5) * 2
        avg = np.mean(data)
        std = np.std(data)
        threshold = 1

        self.assertNotEqual(True, all(data <= avg + threshold * std))  # add assertion here
        cliped = FactorRemoveOutlier.mean_variance_outlier_clip(data, threshold=threshold)

        self.assertEqual(True, all(cliped <= avg + threshold * std))  # add assertion here

    class MyTestCaseFactorRemoveOutlier(unittest.TestCase):
        def test_mad_outlier_clip(self):
            np.random.seed(1)
            data = (np.random.random(size=(100,)) - 0.5) * 2
            median = np.median(data)
            mad = np.abs(data-median).median()
            threshold = 1

            self.assertNotEqual(True, all(data <= median + threshold * mad))  # add assertion here
            cliped = FactorRemoveOutlier.mean_variance_outlier_clip(data, threshold=threshold)

            self.assertEqual(True, all(cliped <= median + threshold * mad))  # add assertion here


if __name__ == '__main__':
    unittest.main()
