import unittest

import numpy as np
from sklearn import metrics
from sklearn.utils._testing import assert_array_almost_equal

from kmeans import KMeans

class TestKMeans(unittest.TestCase):
    def test_implementation(self):
        x = np.array([[0, 0], [0, 1], [4, 0], [4, 1]])
        kmeans = KMeans(2)
        np.random.seed(2020)
        kmeans.fit(x)
        self.assertAlmostEqual(kmeans.inertia_, 1.0)
        self.assertAlmostEqual(metrics.adjusted_rand_score(kmeans.labels_, [0, 0, 1, 1]), 1.0)
        if np.abs(kmeans.cluster_centers_[0, 0] - 4) < 1e-5:
            assert_array_almost_equal(kmeans.cluster_centers_, np.array([[4, 0.5], [0, 0.5]]))
        else:
            assert_array_almost_equal(kmeans.cluster_centers_, np.array([[0, 0.5], [4, 0.5]]))
    def test_gaussian_mixture(self):
        pass

if __name__ == '__main__':
    unittest.main()