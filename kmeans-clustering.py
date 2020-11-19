'''
This file is only a loose template.
You can freely modify any part of it to add more detailed code to support your
analysis.
'''
import numpy as np
from matplotlib import pyplot as plt

# Although using your self-implemented Kmeans is recommended,
# you can use sklearn.cluster.KMeans
# in this experiment.
# If you want to use sklearn.cluster.KMeans,
# please modify the following import correspondingly
from kmeans import KMeans

class data_descriptor(object):
    def __init__(self,data):
        self.data = data
    def plot(self):
        plt.title('data generated from gaussian mixture model')
        plt.scatter(self.data[:,0], self.data[:,1]) 
        plt.show()

class gaussian_mixture_generator(data_descriptor):

    def __init__(self, mean, covariance, weight, data_points = 300 ):
        num_component = len(weight)
        mixture_type_list = list(np.random.choice(np.arange(num_component), size=data_points, p=weight))
        whole_data = np.random.multivariate_normal(mean[0, :], covariance[0, :, :], size=mixture_type_list.count(0))
        for i in range(1, num_component):
            set_tmp = np.random.multivariate_normal(mean[i, :], covariance[i, :, :], size=mixture_type_list.count(i))
            whole_data = np.concatenate((whole_data,set_tmp))
        super(gaussian_mixture_generator, self).__init__(whole_data)

class Kmeans_algorithm(KMeans):
    '''
    kmeans wrapper with plotting support
    '''
    def __init__(self, input_data, n_clusters):
        self.data = input_data
        self.num = self.data.shape[0] # row, n data points
        self.d = input_data[0,:].shape[0]
        super().__init__(n_clusters)
    def fit(self):
        '''
        fit the model with self.data
        '''
        super().fit(self.data)
    def plot(self, savefig_name):
        color_vector = ['r', 'b', 'g', 'm', 'y', 'c', 'k']
        if self.n_clusters >= 7:
            raise NotImplementedError("plot routine for k >=7 is not implemented")
        if self.d != 2: # this function can only be used for d = 2
            raise NotImplementedError("plot routine for dimension larger than 2 is not implemented")
        for i in range(self.n_clusters):
            category_i = np.where(self.labels_ == i)[0]
            plt.scatter(self.data[category_i, 0], self.data[category_i, 1], color=color_vector[i])
        
        plt.savefig(savefig_name)
        plt.show()

if __name__ == '__main__':
    mu_1 = np.array([1, 1])
    mu_2 = np.array([5, 1])
    mu = np.concatenate((mu_1,mu_2)).reshape(2, 2)
    # covariance matrix
    C_1 = np.array([[0.5, 0],[0, 8]])
    C_2 = np.array([[0.5, 0],[0, 8]])
    Cov = np.concatenate((C_1,C_2)).reshape(2, 2, 2)
    
    # mixture weights
    w_1 = 0.5
    w_2 = 1 - w_1
    weight = [w_1,w_2]
    heuristic_num_clusters = 2
    num_of_samples = 1500
    dg_instance = gaussian_mixture_generator(mu, Cov, weight, data_points=num_of_samples)

    kmeans_instance_from_dg = Kmeans_algorithm(dg_instance.data, heuristic_num_clusters)
    kmeans_instance_from_dg.fit()
    kmeans_instance_from_dg.plot('kmeans-clustering.svg')