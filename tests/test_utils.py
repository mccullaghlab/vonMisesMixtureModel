# tests/test_fit.py

import numpy as np
import torch
import pytest

from bvvmmm.utils import fit_with_attempts, component_scan


def generate_synthetic_data(n_samples=100, n_components=2, random_seed=42):
    np.random.seed(random_seed)
    phi = np.concatenate([
        np.random.vonmises(mu=np.pi/4 * i, kappa=5, size=n_samples // n_components)
        for i in range(n_components)
    ])
    psi = np.concatenate([
        np.random.vonmises(mu=-np.pi/4 * i, kappa=5, size=n_samples // n_components)
        for i in range(n_components)
    ])
    data = np.vstack((phi, psi)).T
    return data


def test_fit_with_attempts_improves_likelihood():
    data = generate_synthetic_data(n_samples=200, n_components=2)

    model = fit_with_attempts(data, n_components=2, n_attempts=3, verbose=False)

    assert hasattr(model, 'll'), "Best model does not have log-likelihood." 
    assert model.ll > -np.inf, "Log-likelihood is not finite."


def test_component_scan_basic():
    data = generate_synthetic_data(n_samples=200, n_components=2)
    components = [1, 2]
    ll, aic, bic, icl, best_models = component_scan(data, components, n_attempts=1, train_frac=1.0, verbose=False)
    assert len(ll) == len(components)
    assert len(aic) == len(components)
    assert len(bic) == len(components)
    assert len(icl) == len(components)
    assert len(best_models) == len(components)

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_utils.py"])

