import unittest
import time
import sys

import numpy as np
from sklearn import metrics
from sklearn.utils._testing import assert_array_almost_equal
from sklearn import datasets
from sklearn import cluster
from sklearn.manifold import spectral_embedding
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.feature_extraction import image

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
                                                     centers=[[3, 3], [-3, -3], [3, -3], [-3, 3]],
                                                     cluster_std=1, random_state=0)
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
                                                     centers=[[3, 3], [-3, -3], [3, -3], [-3, 3]],
                                                     cluster_std=1, random_state=0)
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

@unittest.skipIf(SpectralClustering(2).skip, 'skip bonus question')
class TestNormalizedSpectralClustering(unittest.TestCase):
    def test_normalized_embedding(self):
        x = np.array([[1, 0], [0, 1], [3, 0], [4, 1]])
        sc = SpectralClustering(2)
        sc.affinity_matrix_ = sc._get_affinity_matrix(x)

        embedding_features_standard = spectral_embedding(sc.affinity_matrix_, n_components=2,
                                                         norm_laplacian=True, drop_first=False)
        embedding_features = sc._get_embedding(norm_laplacian=True)
        all_one_vector = embedding_features[:, 0] / embedding_features[0, 0]
        assert_array_almost_equal(all_one_vector, np.ones(4))
        second_vector = embedding_features[:, 1] / embedding_features[0, 1]
        second_vector_standard = embedding_features_standard[:, 1] / embedding_features_standard[0, 1]
        assert_array_almost_equal(second_vector, second_vector_standard)
    def test_image_segmentation(self):
        # https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html
        l = 100
        x, y = np.indices((l, l))
        center1 = (28, 24)
        center2 = (40, 50)
        radius1, radius2 = 16, 14
        circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
        circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
        img = circle1 + circle2
        mask = img.astype(bool)
        img = img.astype(float)
        img += 1 + 0.2 * np.random.randn(*img.shape)
        graph = image.img_to_graph(img, mask=mask) # sparse matrix
        graph.data = np.exp(-graph.data / graph.data.std())
        standard_sc = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack',
                                                 affinity='precomputed', random_state=2020)
        standard_sc.fit(graph)
        labels = standard_sc.labels_
        sc = SpectralClustering(n_clusters=2)
        sc.affinity_matrix_ = graph.todense()
        embedding_features = sc._get_embedding(norm_laplacian=True)
        kmeans = cluster.KMeans(n_clusters=2, random_state=2020)
        kmeans.fit(embedding_features)
        self.assertAlmostEqual(metrics.adjusted_rand_score(labels, kmeans.labels_), 1.0)



if __name__=="__main__":
    if len(sys.argv) > 1:
        unittest.main()
    test_obj = unittest.main(exit=False)
    q1 = 2
    q2 = 4
    q3 = 0
    if len(test_obj.result.skipped) == 0:
        q3 = 2.5
    f_or_e = test_obj.result.failures
    f_or_e.extend(test_obj.result.errors)
    for failure in f_or_e:
        if str(failure[0]).find('KMeans') > 0 and q1 > 0:
            q1 -= 1
        elif str(failure[0]).find('NormalizedSpectralClustering') > 0 and q3 > 0:
            q3 -= 1.5
            if q3 < 0:
                q3 = 0
        elif str(failure[0]).find('SpectralClustering') > 0 and q2 > 0:
            q2 -= 1
    print("Your final score of PA3: ", q1 + q2 + q3)
    if len(f_or_e) > 0:
        exit(-1)
