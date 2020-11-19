import unittest
import time

import numpy as np
from sklearn import metrics
from sklearn.utils._testing import assert_array_almost_equal
from sklearn import datasets
from sklearn import cluster
from sklearn import mixture
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import pairwise_kernels

from kmeans import KMeans
from spectral_clustering import SpectralClustering

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
        pos_list, ground_truth = datasets.make_blobs(n_samples=100,
            centers=[[3, 3], [-3, -3], [3, -3], [-3, 3]], cluster_std=1, random_state=0)
        kmeans = KMeans(4)
        standard_kmeans = cluster.KMeans(4, random_state=0)
        np.random.seed(2020)
        kmeans.fit(pos_list)
        standard_kmeans.fit(pos_list)
        self.assertAlmostEqual(metrics.adjusted_rand_score(kmeans.labels_, ground_truth), 1.0)
        self.assertAlmostEqual(kmeans.inertia_, standard_kmeans.inertia_)

class TestSpectralClustering(unittest.TestCase):
    def test_affinity_matrix_implementation(self):
        x = np.array([[0, 0], [0, 1], [4, 0], [4, 1]])
        standard = cluster.SpectralClustering(2)
        standard.fit(x)
        sc = SpectralClustering(2)
        sc.affinity_matrix_ = sc._get_affinity_matrix(x)
        assert_array_almost_equal(sc.affinity_matrix_, standard.affinity_matrix_)

    def test_get_embedding_feature_implementation(self):
        x = np.array([[0, 0], [0, 1], [4, 0], [4, 1]])
        sc = SpectralClustering(2)
        sc.affinity_matrix_ = sc._get_affinity_matrix(x)

        embedding_features_standard = spectral_embedding(sc.affinity_matrix_, n_components=2,
            norm_laplacian=False, drop_first=False)
        embedding_features = sc._get_embedding()
        all_one_vector = embedding_features[:, 0] / embedding_features[0, 0]
        assert_array_almost_equal(all_one_vector, np.ones(4))
        second_vector = embedding_features[:, 1] / embedding_features[0, 1]
        second_vector_standard = embedding_features_standard[:, 1] / embedding_features_standard[0, 1]
        assert_array_almost_equal(second_vector, second_vector_standard)

    def test_gaussian_mixture(self):
        pos_list, ground_truth = datasets.make_blobs(n_samples=100,
            centers=[[3, 3], [-3, -3], [3, -3], [-3, 3]], cluster_std=1, random_state=0)
        sc = SpectralClustering(4, gamma=2.0)
        np.random.seed(2020)
        sc.fit(pos_list)
        self.assertAlmostEqual(metrics.adjusted_rand_score(sc.labels_, ground_truth), 1.0)

    def test_affinity_matrix_implementation_time(self):
        x = np.random.normal(size=[1000, 10])
        start_time = time.time()
        standard = pairwise_kernels(x, metric='rbf', gamma=0.5)
        time_delta_1 = time.time() - start_time

        start_time = time.time()
        sc = SpectralClustering(2, gamma=0.5)        
        affinity_matrix_ = sc._get_affinity_matrix(x)
        time_delta_2 = time.time() - start_time

        self.assertTrue(10 * time_delta_1 > time_delta_2)
        norm_1 = np.linalg.norm(standard)
        norm_2 = np.linalg.norm(affinity_matrix_)
        self.assertAlmostEqual(norm_1, norm_2)

if __name__ == '__main__':
    unittest.main()