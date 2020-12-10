import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import spectral_embedding

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', const=True, type=bool, nargs='?', default=False)
    args = parser.parse_args()
    x_train, y = _generate_three_circle_data()
    if args.plot:
        plt.ion()
    index = 0
    for gamma in np.linspace(1000, 1400):
        affinity_matrix_ = pairwise_kernels(x_train, metric='rbf', gamma=gamma)
        np.fill_diagonal(affinity_matrix_, 0)
        print(np.max(affinity_matrix_))
        embedding_features = spectral_embedding(affinity_matrix_, n_components=3,
                norm_laplacian=False, drop_first=False)
        if args.plot:
            plt.scatter(embedding_features[:, 1], embedding_features[:, 2])
            plt.title('gamma = %.2f' % gamma)
            plt.savefig(os.path.join(save_dir, 'sc-%02d.png' % index))
            plt.pause(0.5)
            plt.clf() # create the video by "ffmpeg -r 4 -i sc-%02d.png -pix_fmt yuv420p output.mp4"
        index += 1
