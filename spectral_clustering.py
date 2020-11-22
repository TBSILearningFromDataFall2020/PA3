import numpy as np

from kmeans import KMeans

class SpectralClustering:
    """
    spectral clustering based on graph Laplacian and k-means
    the graph similarity is computed from rbf kernel

    Parameters:
    -----------
    n_clusters : integer
        the number of clusters and the number of eigenvectors to take

    gamma: double, optional, Kernel coefficient for rbf

    Attributes:
    -----------

    labels_: list
        Labels of each point

    affinity_matrix_ : array-like, shape (n_samples, n_samples)
        Affinity matrix used for clustering.
    """
    def __init__(self, n_clusters, gamma=1.0):
        self.n_clusters = n_clusters
        self.skip = True # modify it to False for bonus question
        self.gamma = gamma

    def train(self, x_train):
        """Receive the input training data, then learn the model.

        Parameters
        ----------
        x_train: np.array, shape (num_samples, num_features)
        Returns
        -------
        None
        """
        self.affinity_matrix_ = self._get_affinity_matrix(x_train)
        embedding_features = self._get_embedding()
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(embedding_features)
        self.labels_ = kmeans.labels_

    def _get_affinity_matrix(self, x_train):
        '''
        construct similarity matrix from the data

        Returns
        -------
        similarity matrix:
            np.array, shape (num_samples, num_samples)
        '''
        # start of your modification
        return np.ones([x_train.shape[0], x_train.shape[0]])
        # end of your modification

    def _get_embedding(self, norm_laplacian=False):
        '''
        get low dimension features from embedded representation of data
        by taking the first k eigenvectors.
        k should be equal to self.n_clusters

        Parameters
        ----------
        norm_laplacian: bool, optional, default=False
            If True, then compute normalized Laplacian.

        Returns
        -------
        embedded feature:
            np.array, shape (num_samples, k)
        '''
        # start of your modification
        return np.ones([self.affinity_matrix_.shape[0], self.n_clusters])
        # end of your modification

    def fit(self, x_train):
        # alias for train
        self.train(x_train)
