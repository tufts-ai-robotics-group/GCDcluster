import numpy as np

from sklearn.mixture import GaussianMixture
import sklearn.mixture._gaussian_mixture as gm

from gcd_cluster.ss_kmeans import ss_kmeans_plusplus


class SSGMM(GaussianMixture):
    def __init__(self, X_labeled, y_labeled, X_unlabeled, n_components):
        self.X_labeled = X_labeled
        self.y_labeled = y_labeled.astype(int)
        self.X_unlabeled = None  # dummy value needed for sklearn validation
        self.n_labeled = X_labeled.shape[0]
        means_init = ss_kmeans_plusplus(
            X_labeled, y_labeled, X_unlabeled, n_components - len(np.unique(y_labeled)))
        super().__init__(n_components, covariance_type="spherical", means_init=means_init)
        self.resp_labeled = np.zeros((self.n_labeled, self.n_components))
        self.resp_labeled[np.arange(self.resp_labeled.shape[0]), self.y_labeled] = 1
        # ignore warning about taking log(0)
        with np.errstate(divide="ignore"):
            self.log_resp_labeled = np.log(self.resp_labeled)

    def _e_step(self, X_unlabeled):
        X = np.vstack((self.X_labeled, X_unlabeled))
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        # force responsibility based on labels
        log_resp[:self.n_labeled] = self.log_resp_labeled
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X_unlabeled, log_resp):
        X = np.vstack((self.X_labeled, X_unlabeled))
        self.weights_, self.means_, self.covariances_ = gm._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = gm._compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
