import itertools

import numpy as np

import gcd_cluster.stats as stats
from gcd_cluster.ss_gmm import SSGMM, DeepSSGMM


def test_ss_gmm():
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

    ss_est = SSGMM(X_labeled, y_labeled, X_unlabeled, n_clusters).fit(X_unlabeled)
    y_pred = ss_est.predict(np.vstack((X_labeled, X_unlabeled)))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    assert acc == 1


def test_deep_ss_gmm():
    rng = np.random.default_rng(0)
    # generate multivariate normal clusters
    n_dim = 2
    n_clusters = 10
    n_samples = 100
    n_labeled = 5
    means = np.array([
        [np.cos(2 * np.pi * i / 10), np.sin(2 * np.pi * i / 10)] for i in range(n_clusters)])
    covariance = 1e-3
    cov = np.eye(n_dim) * covariance
    X_labeled = np.array([
        rng.multivariate_normal(means[i], cov, n_samples) for i in range(n_labeled)
        ]).reshape((-1, n_dim))
    y_labeled = np.array([[i] * n_samples for i in range(n_labeled)]).flatten()
    X_unlabeled = np.array([
        rng.multivariate_normal(means[i], cov, n_samples) for i in range(n_clusters)
        ]).reshape((-1, n_dim))
    y_true = np.array([[i] * n_samples for i in
                       itertools.chain(range(n_labeled), range(n_clusters))]).flatten()

    ss_est = DeepSSGMM(X_labeled, y_labeled, X_unlabeled, n_clusters, covariance)
    for _ in range(ss_est.max_iter):
        print(ss_est.means_)
        resp_unlabeled = ss_est.deep_e_step(X_unlabeled)
        # y_true being used for preds since it has to have all clusters in it to prevent reinit
        ss_est.deep_m_step(X_labeled, y_labeled, X_unlabeled, resp_unlabeled, y_true, covariance)
    y_pred = ss_est.predict(np.vstack((X_labeled, X_unlabeled)))
    row_ind, col_ind, weight = stats.assign_clusters(y_pred, y_true)
    acc = stats.cluster_acc(row_ind, col_ind, weight)
    assert acc == 1
