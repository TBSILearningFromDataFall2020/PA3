'''
extra experiments to explain spectral clustering for toy dataset
'''
import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

save_dir = 'build/gamma/'

def _generate_three_circle_data():
    pos_list = []
    num_list = [60, 100, 140]
    ground_truth = []
    rd = random.Random()
    # make the result reproducible across multiple run
    rd.seed(0)
    for i in range(1, 4): # radius: 0.1 * i
        for _ in range(num_list[i - 1]):
            r = 0.1 * i + 0.01 * (2 * rd.random() - 1)
            angle = 2 * np.pi * rd.random()
            pos_list.append([r * np.cos(angle), r * np.sin(angle)])
            ground_truth.append(i)
    return (np.asarray(pos_list), np.asarray(ground_truth))

def get_the_critical_point(gamma_list, acc_list):
    for i in range(len(gamma_list)):
        if acc_list[i] < 0.99 and acc_list[i + 1] > 0.99:
            return (gamma_list[i], gamma_list[i + 1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', const=True, type=bool, nargs='?', default=False)
    parser.add_argument('--small', const=True, type=bool, nargs='?', default=False)
    args = parser.parse_args()
    x_train, y = _generate_three_circle_data()
    if args.plot:
        plt.ion()
    index = 0
    if args.small:
        gamma_list = np.linspace(1000, 1400)
        acc_list = []
        for gamma in np.linspace(1000, 1400):
            affinity_matrix_ = pairwise_kernels(x_train, metric='rbf', gamma=gamma)
            np.fill_diagonal(affinity_matrix_, 0)
            embedding_features = spectral_embedding(affinity_matrix_, n_components=3,
                    norm_laplacian=False, drop_first=False)
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(embedding_features[:, 1:3])
            acc_list.append(adjusted_rand_score(kmeans.labels_, y))
            if args.plot:
                plt.scatter(embedding_features[:, 1], embedding_features[:, 2])
                plt.title('gamma = %.2f' % gamma)
                plt.savefig(os.path.join(save_dir, 'sc-%02d.png' % index))
                plt.pause(0.5)
                plt.clf() # create the video by "ffmpeg -r 4 -i sc-%02d.png -pix_fmt yuv420p output.mp4"
            index += 1
        g1, g2 = get_the_critical_point(gamma_list, acc_list)
        print('the smallest gamma is between', g1, g2)
