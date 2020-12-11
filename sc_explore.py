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

from spectral_clustering import SpectralClustering

save_dir = 'build/gamma/'
color_vector = ['r', 'b', 'g', 'm', 'y', 'c', 'k']

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
        elif acc_list[i] > 0.99 and acc_list[i + 1] < 0.99:
            return (gamma_list[i], gamma_list[i + 1])
    print(gamma_list, acc_list)
    raise ValueError('no threshold found')

def get_reordered_matrix(affinity_matrix, labels, k=3, enhance=False):
    n = len(labels)
    new_labels = np.zeros(n, dtype=int)
    index = 0
    for i in range(k):
        category_i = np.where(labels == i)[0]
        for i in category_i:
            new_labels[index] = i
            index += 1
    new_affinity_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            new_affinity_matrix[i, j] = affinity_matrix[new_labels[i], new_labels[j]]
    if enhance:
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                new_affinity_matrix[i, j] = (new_affinity_matrix[i - 1, j - 1] + \
                                             new_affinity_matrix[i + 1, j - 1] + \
                                             new_affinity_matrix[i - 1, j + 1] + \
                                             new_affinity_matrix[i + 1, j + 1] ) / 4
    return new_affinity_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', const=True, type=bool, nargs='?', default=False)
    parser.add_argument('--small', const=True, type=bool, nargs='?', default=False)
    parser.add_argument('--gamma', type=float, default=-1)
    parser.add_argument('--num_of_clusters', type=int, default=3)
    args = parser.parse_args()
    x_train, y = _generate_three_circle_data()
    if args.plot:
        plt.ion()
    index = 0
    if args.gamma > 0:
        sc = SpectralClustering(3, gamma=args.gamma)
        sc.affinity_matrix_ = sc._get_affinity_matrix(x_train)
        embedding_features = sc._get_embedding()
        kmeans = KMeans(n_clusters=args.num_of_clusters)
        kmeans.fit(embedding_features[:, 1:args.num_of_clusters])
        np.fill_diagonal(sc.affinity_matrix_, 0)
        new_affinity_matrix = get_reordered_matrix(sc.affinity_matrix_, kmeans.labels_, args.num_of_clusters, False)
        plt.matshow(new_affinity_matrix, cmap='Greys')
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'matrix-%.2f.png' % args.gamma))
        plt.show()
    elif args.small:
        gamma_list = np.linspace(1000, 1400)
        acc_list = []
        inertia_list = []
        for gamma in gamma_list:
            affinity_matrix_ = pairwise_kernels(x_train, metric='rbf', gamma=gamma)
            np.fill_diagonal(affinity_matrix_, 0)
            embedding_features = spectral_embedding(affinity_matrix_, n_components=3,
                    norm_laplacian=False, drop_first=False)
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(embedding_features[:, 1:3])
            acc_list.append(adjusted_rand_score(kmeans.labels_, y))
            inertia_list.append(kmeans.inertia_)
            if args.plot:
                plt.scatter(embedding_features[:, 1], embedding_features[:, 2])
                plt.title('gamma = %.2f' % gamma, fontsize=15)
                plt.savefig(os.path.join(save_dir, 'sc-%02d.png' % index))
                plt.pause(0.5)
                plt.clf() # create the video by "ffmpeg -r 4 -i sc-%02d.png -pix_fmt yuv420p output.mp4"
            index += 1
        g1, g2 = get_the_critical_point(gamma_list, acc_list)
        print('the smallest gamma is between', g1, g2)
        plt.plot(gamma_list, inertia_list)
        plt.title('kmeans inertia varies as gamma increases')
        plt.xlabel('$\gamma$', fontsize=16)
        plt.ylabel('inertia', fontsize=16)
        plt.savefig(os.path.join(save_dir, 'small.png'))
    else:
        gamma_list = np.linspace(9500, 11000)
        acc_list = []
        inertia_list = []
        for gamma in gamma_list:
            sc = SpectralClustering(3, gamma=gamma)
            sc.affinity_matrix_ = sc._get_affinity_matrix(x_train)
            np.fill_diagonal(sc.affinity_matrix_, 0)
            embedding_features = sc._get_embedding()
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(embedding_features[:, 1:3])
            acc = adjusted_rand_score(kmeans.labels_, y)
            acc_list.append(acc)
            inertia_list.append(kmeans.inertia_)
            if args.plot:
                data = embedding_features[:, 1:3]
                for i in range(kmeans.n_clusters):
                    category_i = np.where(kmeans.labels_ == i)[0]
                    plt.scatter(data[category_i, 0], data[category_i, 1], color=color_vector[i])
                plt.title('gamma = %.2f, acc= %.2f, inertia=%.2f' % (gamma, acc, kmeans.inertia_))
                plt.savefig(os.path.join(save_dir, 'big-sc-%02d.png' % index))
                plt.pause(0.5)
                plt.clf() # create the video by "ffmpeg -r 4 -i sc-%02d.png -pix_fmt yuv420p output.mp4"
            index += 1
        g1, g2 = get_the_critical_point(gamma_list, acc_list)
        print('the largest gamma is between', g1, g2)
        plt.plot(gamma_list, inertia_list)
        # plt.title('kmeans inertia varies as gamma increases')
        plt.xlabel('gamma')
        plt.ylabel('inertia')
        plt.savefig(os.path.join(save_dir, 'big.png'))
        plt.clf()
        # plot in the original space
        gamma = g2
        sc = SpectralClustering(3, gamma=gamma)
        sc.affinity_matrix_ = sc._get_affinity_matrix(x_train)
        embedding_features = sc._get_embedding()
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(embedding_features[:, 1:3])
        for i in range(kmeans.n_clusters):
            category_i = np.where(kmeans.labels_ == i)[0]
            plt.scatter(x_train[category_i, 0], x_train[category_i, 1], color=color_vector[i])
        plt.savefig(os.path.join(save_dir, '%.2f.png' % gamma))