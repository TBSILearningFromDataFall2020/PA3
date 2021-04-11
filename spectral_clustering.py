import numpy as np
import networkx as nx

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
        self.skip = False # modify it to False for bonus question
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

        Parameters
        ----------
        x_train: array-like or network graph

        Returns
        -------
        similarity matrix:
            np.array, shape (num_samples, num_samples)
        '''
        # start of your modification
        if isinstance(x_train, nx.Graph):
            return nx.adjacency_matrix(x_train)
        # construct affinity matrix
        n = x_train.shape[0] # num_data
        m = x_train.shape[-1] # num_features
        cross_ = x_train @ x_train.T
        cross_diag = np.diag(cross_)
        all_one_v = np.ones([n])
        square_mat = np.kron(all_one_v, cross_diag).reshape([n, n])
        square_mat += np.kron(cross_diag, all_one_v).reshape([n, n])
        square_mat -= 2 * cross_
        return np.exp(-self.gamma * square_mat)
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
        n = self.affinity_matrix_.shape[0]
        # compute the unnormalized Laplacian
        D = np.sum(self.affinity_matrix_, axis=0)
        L =  np.diag(D) - self.affinity_matrix_
        if norm_laplacian:
            m = np.array(self.affinity_matrix_)
            np.fill_diagonal(m, 0)
            D = np.sum(m, axis=0)
            L = np.eye(D.shape[0]) - np.diag(1.0 / D) @ m
        values, vectors = np.linalg.eig(L)
        Ls = [[i, np.real(values[i])] for i in range(n)]
        Ls.sort(key=lambda x:x[1])
        k = self.n_clusters
        selected_array = [Ls[i][0] for i in range(k)]
        return np.real(vectors[:, selected_array])
        # end of your modification

    def fit(self, x_train):
        # alias for train
        self.train(x_train)
