import unittest

import numpy as np
from sklearn import metrics
from sklearn.utils._testing import assert_array_almost_equal
from sklearn import datasets
from sklearn import cluster

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
        pos_list, ground_truth = datasets.make_blobs(n_samples = 100,
            centers=[[3,3],[-3,-3],[3,-3],[-3,3]], cluster_std=1, random_state=0)
        kmeans = KMeans(4)
        standard_kmeans = cluster.KMeans(4, random_state=0)
        np.random.seed(2020)
        kmeans.fit(pos_list)
        standard_kmeans.fit(pos_list)
        self.assertAlmostEqual(metrics.adjusted_rand_score(kmeans.labels_, ground_truth), 1.0)
        self.assertAlmostEqual(kmeans.inertia_, standard_kmeans.inertia_)

if __name__ == '__main__':
    unittest.main()