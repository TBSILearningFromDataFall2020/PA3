import unittest

import numpy as np
from sklearn import metrics

from kmeans import KMeans

class TestKMeans(unittest.TestCase):
    def test_implementation(self):
        x = np.array([[0, 0], [0, 1], [4, 0], [4, 1]])
        kmeans = KMeans(2)
        np.random.seed(2020)
        kmeans.fit(x)
        self.assertAlmostEqual(metrics.adjusted_rand_score(kmeans.labels_, [0, 0, 1, 1]), 1.0)

if __name__ == '__main__':
    unittest.main()