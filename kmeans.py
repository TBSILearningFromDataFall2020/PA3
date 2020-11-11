import numpy as np
class KMeans:
    """
    Parameters:
    -----------
    n_clusters : the number of clusters to divide the input_data        
    tol: double, optional, the stopping criteria for the loss function
    max_iter: int, optional, the maximal number of iteration        
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
        inner_x_train = x_train.astype('float')
        num_of_data = x_train.shape[0]
        dim = x_train.shape[-1] # dimension of features
        # we take initial centroid to be randomly sampled points
        centroid_index  = np.random.choice(np.arange(num_of_data), size=self.n_clusters, replace=False)
        cluster_centers_  = inner_x_train[centroid_index, :]

        S = np.zeros(num_of_data, dtype=int)
        last_cluster_centers_ = np.zeros([self.n_clusters, dim])
        iteration_cnt = 0
        while np.linalg.norm(cluster_centers_ - last_cluster_centers_) >= self.tol \
            and iteration_cnt < self.max_iter:
            last_cluster_centers_ = cluster_centers_.copy()
            #*********** Assignment ***************
            for i in range(num_of_data):
                S[i] = np.argmin(np.linalg.norm(inner_x_train[i, :] - cluster_centers_, axis=1))
            #*********** Update ******************
            for i in range(self.n_clusters):
                index_list = np.where(S==i)[0]
                if not len(index_list) == 0:
                    cluster_centers_[i, :] = np.mean(inner_x_train[index_list, :], axis=0)
            iteration_cnt += 1
        self.cluster_centers_ = cluster_centers_
        self.labels_ = S
        # compute loss here
        loss = 0
        for i in range(num_of_data):
            current_centroid = cluster_centers_[S[i], :]
            loss += np.linalg.norm(inner_x_train[i, :] - current_centroid) ** 2
        self.inertia_ = loss

    def fit(self, x_train):
        # alias for train
        self.train(x_train)