# tests/test_fit.py

import numpy as np
import torch
import pytest

from bvvmmm.core import SineBVvMMM
from bvvmmm.utils import fit_with_attempts


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


def test_sine_bvvmmm_basic_fit():
    data = generate_synthetic_data(n_samples=200, n_components=2)

    model = SineBVvMMM(n_components=2, max_iter=50, tol=1e-4, verbose=False)
    model.fit(data)

    assert model.weights_ is not None, "Model weights not set."
    assert model.means_ is not None, "Model means not set."
    assert model.kappas_ is not None, "Model kappas not set."

    # Check if the weights sum to 1
    np.testing.assert_allclose(torch.sum(model.weights_).cpu().numpy(), 1.0, rtol=1e-3)

def test_predict_shapes():
    data = generate_synthetic_data(n_samples=100, n_components=2)

    model = SineBVvMMM(n_components=2, max_iter=50, tol=1e-4, verbose=False)
    model.fit(data)

    labels, ll = model.predict(data)
    assert labels.shape[0] == data.shape[0], "Number of labels does not match number of samples."
    assert isinstance(ll, torch.Tensor), "Log-likelihood is not a torch tensor."
    assert ll.dim() == 0, "Log-likelihood should be a scalar."


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_core.py"])

