import numpy as np

class KMeans:
    """
    Parameters:
    -----------
    n_clusters : the number of clusters
    tol: double, optional, the stopping criteria for the loss function
    max_iter: int, optional, the maximal number of iteration

    Attributes
    ----------
    cluster_centers_: array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_: list
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    """
    def __init__(self, n_clusters, tol=1e-4, max_iter=300):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter

    def train(self, x_train):
        """Receive the input training data, then learn the model.
        Parameters
        ----------
        x_train: np.array, shape (num_samples, num_features)
        Returns
        -------
        None
        """
        # modify the following code
        self.cluster_centers_ = np.zeros([self.n_clusters, x_train.shape[-1]])
        self.labels_ = np.zeros(x_train.shape[0], dtype=int)
        self.inertia_ = 0
        # end of your modification

    def fit(self, x_train):
        # alias for train
        self.train(x_train)
