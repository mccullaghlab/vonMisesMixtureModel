# vonMisesMixtureModel

[![PyPI version](https://badge.fury.io/py/bvvmmm.svg)](https://badge.fury.io/py/bvvmmm)
[![Python package](https://github.com/mccullaghlab/vonMisesMixtureModel/actions/workflows/python-package.yml/badge.svg)](https://github.com/mccullaghlab/vonMisesMixtureModel/actions/workflows/python-package.yml)
[![Examples](https://img.shields.io/badge/Examples-Available-blue)](https://github.com/mccullaghlab/vonMisesMixtureModel/tree/main/examples)

A PyTorch-based implementation of an Expectation-Maximization (EM) algorithm for fitting **mixtures of independent Sine Bivariate von Mises (BVM) distributions**. This package is designed for analyzing **circular data**, such as **pairs of phi/psi dihedral angles** in biomolecules or other angular systems.

---

## 📜 Overview

This model fits mixtures of **Bivariate von Mises Sine distributions** on angular data pairs $(\phi, \psi)$ using a flexible and GPU-accelerated EM algorithm.

✅ **Features**:

* EM algorithm for clustering angular data
* Supports **independent phi/psi angle pairs**
* GPU acceleration via PyTorch
* Analytic and numeric M-step updates
* Model scoring: AIC, BIC, ICL
* Visualization tools for fitted models

---

## 📦 Installation

Clone the repo and install:

```bash
git clone https://github.com/mccullaghlab/vonMisesMixtureModel.git
cd vonMisesMixtureModel
pip install .
```

**Dependencies**:

* `torch`
* `numpy`
* `scipy`
* `matplotlib`
* `pytest` (for running tests)

---

## 🧐 Usage

```python
from bvvmmm.core import SineBVvMMM
import numpy as np

# Example: Generate synthetic (phi, psi) data
N = 1000  # number of samples
data = np.random.uniform(-np.pi, np.pi, size=(N, 2))  # synthetic (phi, psi) data

# Initialize model
model = SineBVvMMM(n_components=3, max_iter=100, tol=1e-5, verbose=True)

# Fit the model
model.fit(data)

# Predict cluster assignments
clusters, log_likelihood = model.predict(data)

# Evaluate log-probabilities
log_probs = model.ln_pdf(data)

# Visualize clustering (for 2D data)
model.plot_scatter_clusters(data)
```

---

## 🧠 API Overview

### `SineBVvMMM(...)`

Initialize the mixture model.

| Parameter      | Description                                   |
| -------------- | --------------------------------------------- |
| `n_components` | Number of clusters                            |
| `max_iter`     | Maximum EM iterations                         |
| `tol`          | Convergence threshold for log-likelihood      |
| `device`       | 'cuda' or 'cpu'                               |
| `init_method`  | `random` or `kmeans++` mean initialization    |
| `auto_refine`  | Run numeric `refine()` when coupling is large |
| `small_lambda_rho_thresh` | Threshold for \\(\rho = |\lambda|/\sqrt{\kappa_1\kappa_2}\\) |
| `debug_refine_ratios` | Print per-component \\(\rho\\) values during `fit()` |
| `verbose`      | Print progress during fitting                 |

---

### 🔧 Key Methods

| Method                        | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `fit(data)`                   | Fit model to angular data of shape `(N, 2)`            |
| `predict(data)`               | Predict cluster assignments and compute log-likelihood |
| `ln_pdf(data)`                | Log-density under the fitted model                     |
| `pdf(data)`                   | Probability density under the fitted model             |
| `aic(data)`                   | Akaike Information Criterion                           |
| `bic(data)`                   | Bayesian Information Criterion                         |
| `icl(data)`                   | Integrated Complete Likelihood                         |
| `plot_scatter_clusters(data)` | Visualize 2D clusters                                  |

### 🐞 Refinement troubleshooting

If numeric refinement is triggered unexpectedly often, enable ratio logging:

```python
model = SineBVvMMM(
    n_components=3,
    auto_refine=True,
    small_lambda_rho_thresh=0.30,
    debug_refine_ratios=True,
)
model.fit(data)
```

This prints the per-component ratio \\(\rho = |\lambda|/\sqrt{\kappa_1\kappa_2}\\). Refinement runs only when any component exceeds `small_lambda_rho_thresh`.

---

## 🧬 Applications

* Protein backbone conformational clustering (Ramachandran analysis)
* Directional data clustering (meteorology, geosciences)
* Robotics joint angle analysis
* Wind, wave, or cyclic time series clustering
* Directional statistics in social and behavioral sciences

---

## 🛠️ Testing

To run the unit tests:

```bash
PYTHONPATH=src pytest -q
```

---

## 📚 References

* Mardia & Jupp (2009), *Directional Statistics*
* Boomsma et al. (2008), *Bivariate von Mises for protein geometry*
* Dobson (1978), *Estimating concentration in von Mises distributions*

---

## 🙌 Contributing

Contributions are welcome! Please open an issue or pull request if you'd like to contribute. A `CONTRIBUTING.md` will be added soon.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
