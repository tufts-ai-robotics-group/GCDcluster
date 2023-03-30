import itertools

import numpy as np

import gcd_cluster.stats as stats
from gcd_cluster.ss_kmeans import SSKMeans, ss_kmeans_plusplus


def test_ss_kmeans_plusplus():
    X = np.array([[10, 2], [10, 5], [10, 0]])
    y = np.array([1] * 3)
    X_unlabeled = np.array([[1, 2], [1, 4], [1, 0]])
    centers = ss_kmeans_plusplus(X, y, X_unlabeled, 1)
    # verify labeled center is the mean of the input
    assert np.all(centers[0] == np.mean(X, axis=0))
    # verify unlabeled center is in X_unlabeled
    assert np.any(np.all(centers[1] == X_unlabeled, axis=1))


def test_ss_kmeans():
    rng = np.random.default_rng(0)
    # generate multivariate normal clusters
    n_dim = 2
    n_clusters = 10
    n_samples = 100
    n_labeled = 5
    means = np.array([
        [np.cos(2 * np.pi * i / 10), np.sin(2 * np.pi * i / 10)] for i in range(n_clusters)])
    cov = np.eye(n_dim) * 1e-3
    X_labeled = np.array([
        rng.multivariate_normal(means[i], cov, n_samples) for i in range(n_labeled)
        ]).reshape((-1, n_dim))
    y_labeled = np.array([[i] * n_samples for i in range(n_labeled)]).flatten()
    X_unlabeled = np.array([
        rng.multivariate_normal(means[i], cov, n_samples) for i in range(n_clusters)
        ]).reshape((-1, n_dim))
    y_true = np.array([[i] * n_samples for i in
                       itertools.chain(range(n_labeled), range(n_clusters))]).flatten()

    ss_est = SSKMeans(X_labeled, y_labeled, n_clusters).fit(X_unlabeled)
    y_pred = ss_est.predict(np.vstack((X_labeled, X_unlabeled)))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    assert acc == 1
