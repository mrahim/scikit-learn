"""
Covariance estimators using shrinkage.

Shrinkage corresponds to regularising `cov` using a convex combination:
shrunk_cov = (1-shrinkage)*cov + shrinkage*structured_estimate.

"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD 3 clause

# avoid division truncation
from __future__ import division
import warnings
import numpy as np

from .empirical_covariance_ import empirical_covariance, EmpiricalCovariance
from ..externals.six.moves import xrange
from ..utils import check_array
from ..externals.joblib import Parallel, delayed


# GeneralizedShrunkCovariance estimator

def generalized_shrunk_covariance(emp_cov, shrinkage, structured_estimate
                                  scaling=False):
    """ Calculates a covariance matrix shrunk with structured_estimate
    """
    if scaling:
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features
        return (1 - shrinkage) * emp_cov + shrinkage * mu * structured_estimate
    return (1. - shrinkage) * emp_cov + shrinkage * structured_estimate


class GeneralizedShrunkCovariance(EmpiricalCovariance):
    """Covariance estimator with shrinkage
    """

    def __init__(self, store_precision=True, assume_centered=False,
                 shrinkage=0.1, structured_estimate=None, scaling=False):
        super(GeneralizedShrunkCovariance, self).__init__(
            store_precision=store_precision,
            assume_centered=assume_centered)
        self.shrinkage_ = shrinkage
        self.structured_estimate = structured_estimate
        self.scaling = scaling

    def fit(self, X, y=None):
        """ Fits the shrunk covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_array(X)
        # Not calling the parent object to fit, to avoid a potential
        # matrix inversion when setting the precision
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(
            X, assume_centered=self.assume_centered)
        covariance = generalized_shrunk_covariance(
            covariance, self.shrinkage_,
            self.structured_estimate, self.scaling)
        self._set_covariance(covariance)
        return self


# ShrunkCovariance estimator

def shrunk_covariance(emp_cov, shrinkage=0.1):
    """Calculates a covariance matrix shrunk on the diagonal

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    emp_cov : array-like, shape (n_features, n_features)
        Covariance matrix to be shrunk

    shrinkage : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Returns
    -------
    shrunk_cov : array-like
        Shrunk covariance.

    Notes
    -----
    The regularized (shrunk) covariance is given by

    (1 - shrinkage)*cov
      + shrinkage*mu*np.identity(n_features)

    where mu = trace(cov) / n_features

    """
    emp_cov = check_array(emp_cov)
    n_features = emp_cov.shape[0]

    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    return shrunk_cov


class ShrunkCovariance(EmpiricalCovariance):
    """Covariance estimator with shrinkage

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : boolean, default True
        Specify if the estimated precision is stored

    shrinkage : float, 0 <= shrinkage <= 1, default 0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    `shrinkage` : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized covariance is given by

    (1 - shrinkage)*cov
      + shrinkage*mu*np.identity(n_features)

    where mu = trace(cov) / n_features

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 shrinkage=0.1):
        super(ShrunkCovariance, self).__init__(store_precision=store_precision,
                                               assume_centered=assume_centered)
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        """ Fits the shrunk covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.

        """
        X = check_array(X)
        # Not calling the parent object to fit, to avoid a potential
        # matrix inversion when setting the precision
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = empirical_covariance(
            X, assume_centered=self.assume_centered)
        covariance = shrunk_covariance(covariance, self.shrinkage)
        self._set_covariance(covariance)

        return self


# Ledoit-Wolf estimator

def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : Boolean
        If True, data are not centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data are centered before computation.

    block_size : int
        Size of the blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage: float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage)*cov
      + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
    n_samples, n_features = X.shape

    # optionally center data
    if not assume_centered:
        X = X - X.mean(0)

    # A non-blocked version of the computation is present in the tests
    # in tests/test_covariance.py

    # number of blocks to split the covariance matrix into
    n_splits = int(n_features / block_size)
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = 0.  # sum of the coefficients of <X2.T, X2>
    delta_ = 0.  # sum of the *squared* coefficients of <X.T, X>
    # starting block computation
    for i in xrange(n_splits):
        for j in xrange(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits:]))
        delta_ += np.sum(
            np.dot(X.T[rows], X[:, block_size * n_splits:]) ** 2)
    for j in xrange(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, cols]))
        delta_ += np.sum(
            np.dot(X.T[block_size * n_splits:], X[:, cols]) ** 2)
    delta_ += np.sum(np.dot(X.T[block_size * n_splits:],
                            X[:, block_size * n_splits:]) ** 2)
    delta_ /= n_samples ** 2
    beta_ += np.sum(np.dot(X2.T[block_size * n_splits:],
                           X2[:, block_size * n_splits:]))
    # use delta_ to compute beta
    beta = 1. / (n_features * n_samples) * (beta_ / n_samples - delta_)
    # delta is the sum of the squared coefficients of (<X.T,X> - mu*Id) / p
    delta = delta_ - 2. * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    # get final beta as the min between beta and delta
    # We do this to prevent shrinking more than "1", which whould invert
    # the value of covariances
    beta = min(beta, delta)
    # finally get shrinkage
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage


def ledoit_wolf(X, assume_centered=False, block_size=1000):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    assume_centered : boolean, default=False
        If True, data are not centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data are centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage)*cov
      + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X ** 2).mean()), 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        X, assume_centered=assume_centered, block_size=block_size)
    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp_cov)) / n_features
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage


class LedoitWolf(EmpiricalCovariance):
    """LedoitWolf Estimator

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in "A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices", Ledoit and Wolf, Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float, 0 <= shrinkage <= 1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised covariance is::

        (1 - shrinkage)*cov
                + shrinkage*mu*np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices",
    Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2,
    February 2004, pages 365-411.

    """
    def __init__(self, store_precision=True, assume_centered=False,
                 block_size=1000):
        super(LedoitWolf, self).__init__(store_precision=store_precision,
                                         assume_centered=assume_centered)
        self.block_size = block_size

    def fit(self, X, y=None):
        """ Fits the Ledoit-Wolf shrunk covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.

        """
        # Not calling the parent object to fit, to avoid computing the
        # covariance matrix (and potentially the precision)
        X = check_array(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance, shrinkage = ledoit_wolf(X - self.location_,
                                            assume_centered=True,
                                            block_size=self.block_size)
        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self


# OAS estimator

def oas(X, assume_centered=False):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate.

    assume_centered : boolean
      If True, data are not centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data are centered before computation.

    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularised (shrunk) covariance is:

    (1 - shrinkage)*cov
      + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    The formula we used to implement the OAS
    does not correspond to the one given in the article. It has been taken
    from the MATLAB program available from the author's webpage
    (http://tbayes.eecs.umich.edu/yilun/covestimation).

    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X ** 2).mean()), 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    emp_cov = empirical_covariance(X, assume_centered=assume_centered)
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov ** 2)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else min(num / den, 1.)
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    return shrunk_cov, shrinkage


class OAS(EmpiricalCovariance):
    """Oracle Approximating Shrinkage Estimator

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    OAS is a particular form of shrinkage described in
    "Shrinkage Algorithms for MMSE Covariance Estimation"
    Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

    The formula used here does not correspond to the one given in the
    article. It has been taken from the Matlab program available from the
    authors' webpage (http://tbayes.eecs.umich.edu/yilun/covestimation).
    In the original article, formula (23) states that 2/p is multiplied by
    Trace(cov*cov) in both the numerator and denominator, this operation is omitted
    in the author's MATLAB program because for a large p, the value of 2/p is so
    small that it doesn't affect the value of the estimator.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered: bool, default=False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data are centered before computation.

    Attributes
    ----------
    covariance_ : array-like, shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : array-like, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float, 0 <= shrinkage <= 1
      coefficient in the convex combination used for the computation
      of the shrunk estimate.

    Notes
    -----
    The regularised covariance is::

        (1 - shrinkage)*cov
                + shrinkage*mu*np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the OAS formula (see References)

    References
    ----------
    "Shrinkage Algorithms for MMSE Covariance Estimation"
    Chen et al., IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

    """

    def fit(self, X, y=None):
        """ Fits the Oracle Approximating Shrinkage covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self: object
            Returns self.

        """
        X = check_array(X)
        # Not calling the parent object to fit, to avoid computing the
        # covariance matrix (and potentially the precision)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        covariance, shrinkage = oas(X - self.location_, assume_centered=True)
        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self


# utils function
def _form_symmetric(function, eigenvalues, eigenvectors):
    """Return the symmetric matrix with the given eigenvectors and
    eigenvalues transformed by function.

    Parameters
    ----------
    function : function numpy.ndarray -> numpy.ndarray
        The transform to apply to the eigenvalues.

    eigenvalues : numpy.ndarray, shape (n_features, )
        Input argument of the function.

    eigenvectors : numpy.ndarray, shape (n_features, n_features)
        Unitary matrix.

    Returns
    -------
    output : numpy.ndarray, shape (n_features, n_features)
        The symmetric matrix obtained after transforming the eigenvalues, while
        keeping the same eigenvectors.
    """
    return np.dot(eigenvectors * function(eigenvalues), eigenvectors.T)


# WhitenedLedoitWolf
def whitened_ledoit_wolf(X, structured_estimate, assume_centered=False,
                         block_size=1000, shrink_eigenvalues=False):
    """Estimates the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate

    structured_estimate : array-like, shape (n_features, n_features)
        Prior covariance

    assume_centered : boolean, default=False
        If True, data are not centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data are centered before computation.

    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.

    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.

    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    P^.5.dot.(1 - shrinkage)*(P^-.5.dot.cov.dot.P^-.5)
      + shrinkage * mu * np.identity(n_features)).dot.P^.5

    where mu = trace(cov) / n_features

    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not assume_centered:
            X = X - X.mean()
        return np.atleast_2d((X ** 2).mean()), 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    # whitening goes here
    vals_prior, vecs_prior = np.linalg.eigh(structured_estimate)
    prior_sqrt = _form_symmetric(np.sqrt, vals_prior, vecs_prior)
    prior_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_prior, vecs_prior)

    # whitening wrt inv_prior
    prior_inv_X = np.dot(X, prior_inv_sqrt)

    # get Ledoit-Wolf shrinkage
    shrinkage = ledoit_wolf_shrinkage(
        prior_inv_X, assume_centered=assume_centered, block_size=block_size)
    emp_cov = empirical_covariance(
        prior_inv_X, assume_centered=assume_centered)
    mu = np.sum(np.trace(emp_cov)) / n_features

    """
    # set prior
    vals_prior, vecs_prior = np.linalg.eigh(structured_estimate)
    prior_sqrt = _form_symmetric(np.sqrt, vals_prior, vecs_prior)
    prior_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_prior, vecs_prior)
    # prior_sqrt = np.linalg.sqrtm(structured_estimate)
    # prior_inv_sqrt = np.linalg.inv(prior_sqrt)
    """
    # whitening wrt inv_prior
    whitened_matrix = prior_inv_X.T.dot(prior_inv_X)

    if shrink_eigenvalues:
        u, s, v = np.linalg.svd(whitened_matrix)
        shrunk_s = np.power(s, 1. - shrinkage)
        whitened_matrix = u.dot(np.diag(shrunk_s)).dot(v)

    shrunk_cov = (1. - shrinkage) * whitened_matrix
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    # scale back
    shrunk_cov = prior_sqrt.dot(shrunk_cov).dot(prior_sqrt)

    return shrunk_cov, shrinkage


class WhitenedLedoitWolf(LedoitWolf):
    """Ledoit-Wolf shrinkage with a transformed empirical covariance
    towards a prior (e.g. population averaged covariance)
    """

    def __init__(self, structured_estimate, store_precision=True,
                 assume_centered=False, block_size=1000,
                 shrink_eigenvalues=False):
        super(WhitenedLedoitWolf, self).__init__(
            store_precision=store_precision,
            assume_centered=assume_centered)
        self.block_size = block_size
        self.structured_estimate = structured_estimate
        self.shrink_eigenvalues = shrink_eigenvalues

    def fit(self, X, y=None):
        """ Fits the Ledoit-Wolf shrunk covariance model
        according to the given training data, prior, and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.

        """
        # Not calling the parent object to fit, to avoid computing the
        # covariance matrix (and potentially the precision)
        X = check_array(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)

        # whitening goes here
        vals_prior, vecs_prior = np.linalg.eigh(self.structured_estimate)
        prior_sqrt = _form_symmetric(np.sqrt, vals_prior, vecs_prior)
        prior_inv_sqrt = _form_symmetric(np.sqrt, 1. / vals_prior, vecs_prior)

        # whitening wrt inv_prior
        prior_inv_X = np.dot(X - self.location_, prior_inv_sqrt)
        # whitened_matrix = prior_inv_X.dot(prior_inv_X.T)

        covariance, shrinkage = ledoit_wolf(
            prior_inv_X,
            assume_centered=True,
            block_size=self.block_size)

        # scale-back
        covariance = prior_sqrt.dot(covariance).dot(prior_sqrt)
        """
        covariance, shrinkage = whitened_ledoit_wolf(
            X - self.location_, self.structured_estimate,
            shrink_eigenvalues=self.shrink_eigenvalues)
        """

        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self


def score_covariance(Xtrain, Xtest, shrinkage, structured_estimate, metric,
                     scaling):
    gsc = GeneralizedShrunkCovariance(
        shrinkage=shrinkage,
        structured_estimate=structured_estimate
        scaling=scaling)
    gsc.fit(Xtrain)
    if metric == 'mse':
        return gsc.error_norm(empirical_covariance(Xtest))
    else:
        return gsc.score(Xtest)


# GSC-CV
class GeneralizedShrunkCovarianceCV:
    def __init__(self, structured_estimate,
                 shrinkages=np.linspace(0, 1, 11),
                 assume_centered=True,
                 metric='loglikelihood',
                 scaling=False,
                 n_jobs=1):
        self.structured_estimate = structured_estimate
        self.shrinkages = shrinkages
        self.n_jobs = n_jobs
        self.metric = metric
        self.scaling = scaling

    def cross_validation(self, Xtrain, Xtest):
        cv_scores = Parallel(n_jobs=self.n_jobs)(
            delayed(score_covariance)
            (Xtrain, Xtest, shrinkage, self.structured_estimate,
             self.metric, self.scaling)
            for shrinkage in self.shrinkages)
        self.cv_scores_ = cv_scores
        if self.metric == 'mse':
            self.best_score_ = np.min(cv_scores)
            self.shrinkage_ = self.shrinkages[np.argmin(cv_scores)]
        else:
            self.best_score_ = np.max(cv_scores)
            self.shrinkage_ = self.shrinkages[np.argmax(cv_scores)]
        return self.best_score_
