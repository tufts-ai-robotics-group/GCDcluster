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
        # initialize from SS K-means++
        means_init = ss_kmeans_plusplus(
            X_labeled, y_labeled, X_unlabeled, n_components - len(np.unique(y_labeled)))
        super().__init__(n_components, covariance_type="spherical", means_init=means_init)
        # cache labeled responsibilities
        self.resp_labeled = targets_to_resp(y_labeled, self.n_components)
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


class DeepSSGMM(GaussianMixture):
    def __init__(self, X_labeled, y_labeled, X_unlabeled, n_components, covariance: float):
        # dummy value needed for sklearn validation
        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.covariance = None
        # initialize from SS K-means++
        means_init = ss_kmeans_plusplus(
            X_labeled, y_labeled, X_unlabeled, n_components - len(np.unique(y_labeled)))
        super().__init__(n_components, covariance_type="spherical", means_init=means_init)
        self._initialize_parameters(X_unlabeled, np.random.mtrand._rand)
        # force spherical covariance used by model
        self.set_covariance(covariance)

    def deep_e_step(self, X_unlabeled):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X_unlabeled)
        return np.exp(log_resp)

    def deep_m_step(self, X_labeled, y_labeled, X_unlabeled, resp_unlabeled, covariance: float):
        # calculate and concat inputs
        resp_labeled = targets_to_resp(y_labeled, self.n_components)
        X = np.vstack((X_labeled, X_unlabeled))
        resp = np.vstack((resp_labeled, resp_unlabeled))
        # update mixing weights and means, fixing covarinace before and after
        self.set_covariance(covariance)
        self.weights_, self.means_, self.covariances_ = gm._estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.set_covariance(covariance)

    def set_covariance(self, covariance: float):
        self.covariances_ = np.array([covariance] * self.n_components)
        self.precisions_cholesky_ = gm._compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _e_step(self, X_unlabeled):
        raise Exception("E-step disabled for DeepSSGMM, use SSGMMM for fitting")

    def _m_step(self, X_unlabeled, log_resp):
        raise Exception("M-step disabled for DeepSSGMM, use SSGMMM for fitting")


def targets_to_resp(y_labeled, n_components):
    n_labeled = len(y_labeled)
    resp = np.zeros((n_labeled, n_components))
    resp[np.arange(n_labeled), y_labeled] = 1
    return resp
