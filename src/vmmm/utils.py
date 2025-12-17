import numpy as np
import torch
import sys
import warnings
from .core import VonMisesMixture

def fit_with_attempts(data, n_components, n_attempts, verbose=True, max_iter=200, tol=1e-5, device=None, dtype=torch.float64):
    """
    Repeatedly fit a Von Mises Mixture model to data with multiple random initializations,
    and return the model with the highest log-likelihood.

    Parameters
    ----------
    data : ndarray or tensor
        The input angular data.
    n_components : int
        Number of mixture components.
    n_attempts : int
        Number of fitting attempts (with different random initializations).
    verbose : boolean, default=True
        whether to print output information to stdout
    max_iter : int, default=200
        The maximum number of EM iterations allowed
    tol : float, default=1e-5
        The convergence tolerance for log-likelihood improvement. The EM 
        algorithm stops if the change in log-likelihood is smaller than this value.

    Returns
    -------
    best_model : VonMisesMixture
        The model instance with the highest log-likelihood.
    """
    models = []                     # To store all fitted models
    ll = np.empty(n_attempts)       # To store the log-likelihoods

    if verbose == True:
        print(f"{'Attempt':>8} | {'Log-Likelihood':>15}")
        print("-" * 28)

    for i in range(n_attempts):
        # Initialize and fit model
        model = VonMisesMixture(n_components=n_components, max_iter=max_iter, tol=tol)#, device=device, dtype=dtype)
        model.fit(data)

        # Store model and its log-likelihood
        models.append(model)
        ll[i] = model.score(data)
        if verbose == True:
            # Print formatted result for this attempt
            print(f"{i + 1:>8} | {ll[i]:>15.6f}")

    # Return the model with the highest log-likelihood
    return models[np.nanargmax(ll)]


def component_scan(data, components, n_attempts=15, tol=1e-4, train_frac=1.0, verbose=True, device=None, dtype=torch.float64):
    """
    Scan through different numbers of components by fitting multiple attempts
    of the von Mises Mixture model and returning metrics including training log likelihood,
    AIC, BIC, ICL, and (if using a validation split) the cross validation log likelihood.

    Parameters
    ----------
    data : array-like, shape (n_data_points, 1)
        Input data for fitting.
    components : array-like
        A list or array of component counts to test.
    n_attempts : int, default=15
        Number of random initializations to try for each component count.
    tol : float, default=1e-4
        log likelihood tolerance for convergence
    train_frac : float, default=1.0
        Fraction of the data to use for training. If less than 1.0, the remainder
        is held out as a cross validation (CV) set.

    Returns
    -------
    ll : numpy.ndarray
        Best (maximum) training log likelihood for each tested component count.
    aic : numpy.ndarray
        AIC value corresponding to the best training log likelihood.
    bic : numpy.ndarray
        BIC value corresponding to the best training log likelihood.
    icl : numpy.ndarray
        ICL value corresponding to the best training log likelihood.
    cv_ll : numpy.ndarray (if train_frac < 1.0)
        Cross validation log likelihood for each component count.
    best_models : list of best models
    """
    import numpy as np
    import torch

    n_total = data.shape[0]
    if train_frac < 1.0:
        n_train = int(train_frac * n_total)
        # Randomly permute indices and split
        permutation = np.random.permutation(n_total)
        train_data = data[permutation[:n_train]]
        cv_data = data[permutation[n_train:]]
        print(f"Training on {n_train} samples and validating on {n_total - n_train} samples.")
    else:
        train_data = data
        cv_data = None

    n_comp = len(components)
    ll = np.empty(n_comp)
    aic = np.empty(n_comp)
    bic = np.empty(n_comp)
    icl = np.empty(n_comp)
    if cv_data is not None:
        cv_ll = np.empty(n_comp)
    best_models = []
    for i, comp in enumerate(components):
        temp_ll = np.empty(n_attempts)
        if cv_data is not None:
            temp_cv_ll = np.empty(n_attempts)
        models = []
        # no need for more than 1 attempt for components=1
        if comp == 1:
            attempt = 0
            model = VonMisesMixture(n_components=comp, max_iter=200, tol=tol)#, device=device, dtype=dtype)
            # Fit using the training set only
            model.fit(train_data)
            models.append(model)
            temp_ll[attempt] = model.score(train_data)
            if cv_data is not None:
                # Evaluate CV log likelihood on the held-out set.
                # Note: model.predict returns (cluster_ids, log_likelihood)
                temp_cv_ll[attempt] = model.score(cv_data)
            if verbose == True:
                print(f"Components: {comp}, Attempt: {attempt+1}, Training LL: {temp_ll[attempt]}, " +
                  (f"CV LL: {temp_cv_ll[attempt]}" if cv_data is not None else ""))
            best_index = 0
        else:
            for attempt in range(n_attempts):
                model = VonMisesMixture(n_components=comp, max_iter=200, tol=tol)
                # Fit using the training set only
                model.fit(train_data)
                models.append(model)
                temp_ll[attempt] = model.score(train_data)
                if cv_data is not None:
                    # Evaluate CV log likelihood on the held-out set.
                    # Note: model.predict returns (cluster_ids, log_likelihood)
                    temp_cv_ll[attempt] = model.score(cv_data)
                if verbose == True:
                    print(f"Components: {comp}, Attempt: {attempt+1}, Training LL: {temp_ll[attempt]}, " +
                      (f"CV LL: {temp_cv_ll[attempt]}" if cv_data is not None else ""))
            # Choose the attempt with the highest training log likelihood
            best_index = np.nanargmax(temp_ll)
        ll[i] = temp_ll[best_index]
        best_model = models[best_index]
        best_models.append(best_model)
        # Compute the information criteria on the full data (or you could use train_data if preferred)
        #aic[i] = best_model.aic(data)
        #bic[i] = best_model.bic(data)
        #icl[i] = best_model.icl(data)
        if cv_data is not None:
            cv_ll[i] = temp_cv_ll[best_index]

    if cv_data is not None:
        return ll, cv_ll, best_models
        #return ll, aic, bic, icl, cv_ll, best_models
    else:
        return ll, best_models
    #return ll, aic, bic, icl, best_models


