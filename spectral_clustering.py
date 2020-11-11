import numpy as np

from kmeans import KMeans

class SpectralClustering:
    """
    spectral clustering based on unnormalized graph Laplacian and k-means
    the graph similarity is computed from rbf kernel
    Parameters:
    -----------
    n_clusters : the number of clusters and the number of eigenvectors to take
    gamma: double, optional, Kernel coefficient for rbf       
    """
    def __init__(self, n_clusters, gamma=1.0):
        self.n_clusters = n_clusters
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

    def _get_embedding(self):
        n = self.affinity_matrix_.shape[0]
        # compute the unnormalized Laplacian
        L = np.diag(np.sum(self.affinity_matrix_, axis=0))
        L -= self.affinity_matrix_
        values, vectors = np.linalg.eig(L)
        Ls = [[i, values[i]] for i in range(n)]
        Ls.sort(key=lambda x:x[1])
        k = self.n_clusters
        selected_array = [Ls[i][0] for i in range(k)]
        return vectors[:, selected_array]

    def fit(self, x_train):
        # alias for train
        self.train(x_train)