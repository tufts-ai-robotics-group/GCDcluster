import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_common import _inertia_dense, _inertia_sparse, _is_same_clustering
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum, row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


# modified version of SKLearn algorithms
class SSKMeans(KMeans):
    def __init__(self, X_labeled, y, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, verbose=0,
                 random_state=0, copy_x=True):
        super().__init__(n_clusters, n_init=n_init, max_iter=max_iter, tol=tol,
                         verbose=verbose, random_state=random_state, copy_x=copy_x)
        self.X_labeled = self._validate_data(
            X_labeled,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )
        self.y = np.array(y).astype(int)
        self.n_clusters_unlabeled = self.n_clusters - len(np.unique(self.y))
        self.sample_weight_labeled = np.ones_like(self.X_labeled[:, 0])

    def fit(self, X_unlabeled, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X_unlabeled : {array-like, sparse matrix} of shape (n_samples, n_features)
                      Training instances to cluster. It must be noted that the data
                      will be converted to C ordering, which will cause a memory
                      copy if the given data is not C-contiguous.
                      If a sparse matrix is passed, a copy will be made if it's not in
                      CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            .. versionadded:: 0.20
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_unlabeled = self._validate_data(
            X_unlabeled,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X_unlabeled, dtype=X_unlabeled.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X_unlabeled.dtype, copy=True, order="C")
            self._validate_center_shape(X_unlabeled, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X_unlabeled):
            X_mean = np.vstack([self.X_labeled, X_unlabeled]).mean(axis=0)
            # The copy was already done above
            X_unlabeled -= X_mean
            self.X_labeled -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X_unlabeled, squared=True)

        self._check_mkl_vcomp(X_unlabeled, X_unlabeled.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self.n_init):
            # Initialize centers
            centers_init = ss_kmeans_plusplus(
                self.X_labeled, self.y, X_unlabeled, self.n_clusters_unlabeled,
                x_squared_norms=x_squared_norms, random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = self._ss_kmeans_single_lloyd(
                X_unlabeled,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self.tol,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X_unlabeled):
            if not self.copy_x:
                X_unlabeled += X_mean
                self.X_labeled += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters_unlabeled:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters_unlabeled),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def _ss_kmeans_single_lloyd(
        self,
        X_unlabeled,
        sample_weight,
        centers_init,
        max_iter=300,
        verbose=False,
        tol=1e-4,
        n_threads=1,
    ):
        """A single run of semi-supervised k-means lloyd, assumes preparation completed prior.
        Parameters
        ----------
        X_unlabeled : {ndarray, sparse matrix} of shape (n_samples, n_features)
                      The observations to cluster. If sparse matrix, must be in CSR format.
        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.
        centers_init : ndarray of shape (n_clusters, n_features)
            The initial centers.
        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm to run.
        verbose : bool, default=False
            Verbosity mode
        tol : float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference
            in the cluster centers of two consecutive iterations to declare
            convergence.
            It's not advised to set `tol=0` since convergence might never be
            declared due to rounding errors. Use a very small number instead.
        n_threads : int, default=1
            The number of OpenMP threads to use for the computation. Parallelism is
            sample-wise on the main cython loop which assigns each sample to its
            closest center.
        Returns
        -------
        centroid : ndarray of shape (n_clusters, n_features)
            Centroids found at the last iteration of k-means.
        label : ndarray of shape (n_samples,)
            label[i] is the code or index of the centroid the
            i'th observation is closest to.
        inertia : float
            The final value of the inertia criterion (sum of squared distances to
            the closest centroid for all observations in the training set).
        n_iter : int
            Number of iterations run.
        """
        n_clusters = centers_init.shape[0]

        # Buffers to avoid new allocations at each iteration.
        centers = centers_init
        centers_new = np.zeros_like(centers)
        labels = np.full(X_unlabeled.shape[0], -1, dtype=np.int32)
        labels_old = labels.copy()
        weight_in_clusters = np.zeros(n_clusters, dtype=X_unlabeled.dtype)
        center_shift = np.zeros(n_clusters, dtype=X_unlabeled.dtype)

        if sp.issparse(X_unlabeled):
            lloyd_iter = lloyd_iter_chunked_sparse
            _inertia = _inertia_sparse
        else:
            lloyd_iter = lloyd_iter_chunked_dense
            _inertia = _inertia_dense

        strict_convergence = False

        # calculate supervised data contribution to centers
        targets = np.unique(self.y)
        centers_labeled = np.zeros((len(targets), self.X_labeled.shape[1]),
                                   dtype=self.X_labeled.dtype)
        weight_in_clusters_labeled = np.zeros((len(targets),), dtype=self.X_labeled.dtype)
        for i, target in enumerate(targets):
            centers_labeled[i] = np.sum(self.X_labeled[self.y == target], axis=0)
            weight_in_clusters_labeled[i] = np.sum(self.y == target)

        # Threadpoolctl context to limit the number of threads in second level of
        # nested parallelism (i.e. BLAS) to avoid oversubsciption.
        with threadpool_limits(limits=1, user_api="blas"):
            for i in range(max_iter):
                lloyd_iter(
                    X_unlabeled,
                    sample_weight,
                    centers,
                    centers_new,
                    weight_in_clusters,
                    labels,
                    center_shift,
                    n_threads,
                    False  # do not update centers, done manually
                )

                # update centers manually from unsupervised data
                weight_in_clusters.fill(0)
                centers_new.fill(0)
                for label in np.unique(labels):
                    label_mask = labels == label
                    weight_in_clusters[label] += np.sum(sample_weight[label_mask], axis=0)
                    centers_new[label] += np.sum(
                        X_unlabeled[label_mask] * sample_weight[label_mask, np.newaxis], axis=0)
                # add supervised data to clusters
                for j in range(len(targets)):
                    weight_in_clusters[j] += weight_in_clusters_labeled[j]
                    centers_new[j] += centers_labeled[j]
                centers_new = centers_new / weight_in_clusters[:, np.newaxis]

                if verbose:
                    inertia = _inertia(
                        np.vstack((self.X_labeled, X_unlabeled)),
                        np.hstack((self.sample_weight_labeled, sample_weight)),
                        centers, np.hstack((self.y, labels)), n_threads)
                    print(f"Iteration {i}, inertia {inertia}.")

                centers, centers_new = centers_new, centers

                if np.array_equal(labels, labels_old):
                    # First check the labels for strict convergence.
                    if verbose:
                        print(f"Converged at iteration {i}: strict convergence.")
                    strict_convergence = True
                    break
                else:
                    # No strict convergence, check for tol based convergence.
                    center_shift_tot = (center_shift ** 2).sum()
                    if center_shift_tot <= tol:
                        if verbose:
                            print(
                                f"Converged at iteration {i}: center shift "
                                f"{center_shift_tot} within tolerance {tol}."
                            )
                        break

                labels_old[:] = labels

            if not strict_convergence:
                # rerun E-step so that predicted labels match cluster centers
                lloyd_iter(
                    X_unlabeled,
                    sample_weight,
                    centers,
                    centers,
                    weight_in_clusters,
                    labels,
                    center_shift,
                    n_threads,
                    update_centers=False,
                )

        inertia = _inertia(
            np.vstack((self.X_labeled, X_unlabeled)),
            np.hstack((self.sample_weight_labeled, sample_weight)),
            centers, np.hstack((self.y, labels)).astype(np.int32), n_threads)

        return labels, inertia, centers, i + 1


def ss_kmeans_plusplus(
    X_labeled, y, X_unlabeled, n_clusters_unlabeled, *, x_squared_norms=None, random_state=0,
    n_local_trials=None
):
    """Init n_clusters seeds according to semi-supervised k-means++
    Parameters
    ----------
    X_labeled : {array-like, sparse matrix} of shape (n_samples, n_features)
                The labeled data to pick seeds from.
    y : {array-like, sparse matrix} of shape (n_samples,)
        The labels for X.
    X_unlabeled : {array-like, sparse matrix} of shape (n_samples, n_features)
        The unlabeled data to pick seeds from.
    n_clusters_unlabeled : int
        The number of centroids to initialize from unlabeled data.
    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.
    random_state : int or RandomState instance, default=0
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    """
    # Check data
    X_labeled = check_array(X_labeled, accept_sparse="csr", dtype=[np.float64, np.float32])
    X_unlabeled = check_array(X_unlabeled, accept_sparse="csr", dtype=[np.float64, np.float32])

    n_unlabeled = X_unlabeled.shape[0]
    if n_unlabeled < n_clusters_unlabeled:
        raise ValueError(
            f"n_samples={n_unlabeled} should be >= n_clusters_unlabled={n_clusters_unlabeled}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X_unlabeled, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X_labeled.dtype, ensure_2d=False)

    if x_squared_norms.shape[0] != X_unlabeled.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {n_unlabeled}."
        )

    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError(
            f"n_local_trials is set to {n_local_trials} but should be an "
            "integer value greater than zero."
        )

    random_state = check_random_state(random_state)

    # Call private semi-supervised k-means++
    centers = _ss_kmeans_plusplus(
        X_labeled, y, X_unlabeled, n_clusters_unlabeled, x_squared_norms, random_state,
        n_local_trials
    )

    return centers


def _ss_kmeans_plusplus(
    X_labeled, y, X_unlabeled, n_clusters_unlabled, x_squared_norms, random_state,
    n_local_trials=None
):
    """Computational component for initialization of n_clusters by semi-supervised
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X_labeled : {array-like, sparse matrix} of shape (n_samples, n_features)
                The labeled data to pick seeds from.
    y : {array-like, sparse matrix} of shape (n_samples,)
        The labels for X.
    X_unlabeled : {array-like, sparse matrix} of shape (n_samples, n_features)
        The unlabeled data to pick seeds from.
    n_clusters_unlabled : int
        The number of centroids to initialize from unlabeled data.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point in X_unlabeled.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.
    """
    _, n_features = X_labeled.shape

    centers_unlabeled = np.empty((n_clusters_unlabled, n_features), dtype=X_labeled.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters_unlabled))

    # Calculate centers for labeled points
    targets = np.unique(y)
    centers_labeled = np.empty((len(targets), n_features), dtype=X_labeled.dtype)
    for i, target in enumerate(targets):
        centers_labeled[i] = np.mean(X_labeled[y == target], axis=0)

    # Initialize list of closest distances and calculate current potential
    closest_dist_sqs = euclidean_distances(
        centers_labeled, X_unlabeled, Y_norm_squared=x_squared_norms, squared=True
    )
    closest_dist_sq = np.min(closest_dist_sqs, axis=0)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters_unlabled points
    for c in range(n_clusters_unlabled):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X_unlabeled[candidate_ids], X_unlabeled, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X_unlabeled):
            centers_unlabeled[c] = X_unlabeled[best_candidate].toarray()
        else:
            centers_unlabeled[c] = X_unlabeled[best_candidate]

    return np.vstack((centers_labeled, centers_unlabeled))
