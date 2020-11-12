import numpy as np
from scipy import stats

class GMM:
    """
    Gaussian Mixture Model
    Parameters:
    -----------
    n_components : the number of mixture components
    tol: double, optional, the stopping criteria for the loss function
    max_iter: int, optional, the maximal number of iteration        
    """
    def __init__(self, n_components, tol=1e-4, max_iter=300):
        self.n_components = n_components
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
        # see https://github.com/zhaofeng-shu33/signal_processing_experiment/blob/master/simulation_0/EMGMM.m
        # for implementation reference
        # initialize the model parameter randomly
        n_components = self.n_components
        n_data = x_train.shape[0]
        n_features = x_train.shape[-1]

        means_ = np.random.random([n_components, n_features])
        weights_ = np.random.uniform(size=[n_components])
        # constraint for weights
        weights_ /= np.sum(weights_)
        covariances_ = np.zeros([n_components, n_features, n_features])
        # make covariance the identity matrix for each component
        for i in range(n_components):
            covariances_[i] = np.eye(n_features)
        gamma = np.zeros([n_components, n_data])
        old_log_likelihood = 0
        for _ in range(self.max_iter):
            # enter E step
            # first compute the hidden variable
            for i in range(n_components):
                gamma[i, :] = weights_[i] * stats.multivariate_normal.pdf(x_train,
                        mean=means_[i, :], cov=covariances_[i, :])
            norm_sum = np.sum(gamma, axis=0)
            # before normalization, estimate the log probability
            log_likelihood = np.mean(np.log(norm_sum))
            if np.abs(log_likelihood - old_log_likelihood) < self.tol:
                break
            else:
                old_log_likelihood = log_likelihood
            gamma /= norm_sum # each column of gamma must sum to one
            gamma_sum = np.sum(gamma, axis=1)
            # enter M step
            # estimate the parameter
            weights_ = gamma_sum / np.sum(gamma_sum)
            for i in range(n_components):
                means_[i, :] = gamma[i, :] @ x_train / gamma_sum[i]
                tmp = x_train - means_[i, :]
                covariances_[i, :] = tmp.T @ np.diag(gamma[i, :]) @ tmp / gamma_sum[i]
        self.means_ = means_
        self.covariances_ = covariances_
        self.weights_ = weights_
        self.lower_bound_ = log_likelihood

    def fit(self, x_train):
        # alias for train
        self.train(x_train)

    def predict(self, x_test):
        n_data = x_test.shape[0]
        gamma = np.zeros([self.n_components, n_data])
        for i in range(self.n_components):
            gamma[i, :] = self.weights_[i] * stats.multivariate_normal.pdf(x_test,
                            mean=self.means_[i, :], cov=self.covariances_[i, :])
        return gamma.argmax(axis=0)