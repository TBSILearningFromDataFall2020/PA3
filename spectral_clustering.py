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
        m = x_train.shape[-1]
        # construct affinity matrix
        cross_ = x_train @ x_train.T
        cross_diag = np.diag(cross_)
        all_one_v = np.ones([m])
        square_mat = np.kron(all_one_v, cross_diag).reshape([m, m])
        square_mat += np.kron(cross_diag, all_one_v).reshape([m, m])
        square_mat -= 2 * cross_
        affinity_matrix = np.exp(-self.gamma * square_mat)
        
    def fit(self, x_train):
        # alias for train
        self.train(x_train)