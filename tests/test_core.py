# tests/test_fit.py

import numpy as np
import torch
import pytest

from bvvmmm.core import SineBVvMMM


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

    labels = model.predict(data)
    assert labels.shape[0] == data.shape[0], "Number of labels does not match number of samples."


def test_score_shapes():
    data = generate_synthetic_data(n_samples=100, n_components=2)

    model = SineBVvMMM(n_components=2, max_iter=50, tol=1e-4, verbose=False)
    model.fit(data)

    ll = model.score(data)
    assert isinstance(ll, torch.Tensor), "Log-likelihood is not a torch tensor."
    assert ll.dim() == 0, "Log-likelihood should be a scalar."

def test_refine_macro3_res12_stays_finite():
    data = np.load("tests/data/repro_macro3_res12_phi_psi.npy")
    np.random.seed(12)
    torch.manual_seed(12)
    model = SineBVvMMM(
        n_components=3,
        max_iter=3,
        tol=1e-5,
        auto_refine=False,
        verbose=False,
    )
    model.fit(data)
    assert torch.isfinite(model.weights_).all()
    assert torch.isfinite(model.means_).all()
    assert torch.isfinite(model.kappas_).all()
    assert torch.isfinite(model.ll)

    model.refine(data)
    assert torch.isfinite(model.weights_).all()
    assert torch.isfinite(model.means_).all()
    assert torch.isfinite(model.kappas_).all()
    assert torch.isfinite(model.ll)


def test_configurational_entropy_uses_generated_log_probabilities(monkeypatch):
    model = SineBVvMMM(n_components=1, max_iter=1, tol=1e-4, verbose=False)
    model.weights_ = torch.tensor([1.0], dtype=model.dtype, device=model.device)
    model.means_ = torch.tensor([[0.0, 0.0]], dtype=model.dtype, device=model.device)
    model.kappas_ = torch.tensor([[1.0, 1.0, 0.0]], dtype=model.dtype, device=model.device)
    model.normalization_ = model._calculate_normalization_constant(model.kappas_)

    samples = np.array([
        [0.0, 0.0],
        [0.25, -0.25],
        [-0.5, 0.5],
        [1.0, -1.0],
    ])

    def fake_generate(n_points):
        assert n_points == samples.shape[0]
        return samples, np.zeros(n_points, dtype=int)

    monkeypatch.setattr(model, "generate", fake_generate)

    entropy, stderr = model.configurational_entropy(samples.shape[0])
    ln_prob = model.ln_pdf(samples)

    np.testing.assert_allclose(entropy, (-torch.mean(ln_prob)).cpu().numpy())
    np.testing.assert_allclose(
        stderr,
        (torch.std(ln_prob, unbiased=True) / np.sqrt(samples.shape[0])).cpu().numpy(),
    )


def test_configurational_entropy_rejects_non_positive_n_points():
    model = SineBVvMMM(n_components=1, max_iter=1, tol=1e-4, verbose=False)

    with pytest.raises(ValueError, match="n_points must be a positive integer"):
        model.configurational_entropy(0)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_core.py"])
