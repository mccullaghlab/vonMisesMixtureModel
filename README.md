# vonMisesMixtureModel

A PyTorch-based implementation of a flexible Expectation-Maximization (EM) algorithm for fitting **mixtures of products of independent Sine Bivariate von Mises (BVM) distributions**. Designed for analyzing **circular data**, such as **multiple pairs of phi/psi dihedral angles** in biomolecules or other angular systems.

---

## 📘 Overview

This model generalizes the standard BVM mixture model to support **multiple independent bivariate angle pairs**, modeling the **joint probability** as a product of independent BVM distributions. It is ideal for problems where circular data (e.g., torsion angles) occur in structured pairs.

### ✨ Features

- 🔁 EM algorithm for clustering angular data
- 🌀 Models **multiple phi/psi angle pairs**
- 🔗 Assumes angle pairs are **independent across sites**
- ⚡ GPU acceleration with PyTorch
- 📈 Tools for prediction, scoring, and visualization
- 🧪 Ready for integration into workflows for **protein modeling**, **robotics**, **directional statistics**, etc.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/mccullaghlab/vonMisesMixtureModel.git
cd vonMisesMixtureModel
pip install setup.py
```

**Dependencies**:
- `torch`
- `scipy`
- `matplotlib`
- `numpy`

---

## 🧠 Usage

```python
from multi_bvvmmm import MultiSineBVVMMM
import numpy as np

# Example: 1000 samples, 10 pairs of phi/psi angles
N = 1000
M = 10  # number of features (phi/psi pairs)
data = np.random.uniform(-np.pi, np.pi, size=(N, M, 2))  # synthetic data

# Initialize model
model = MultiSineBVVMMM(n_components=3, n_features=M, max_iter=100, verbose=True)

# Fit the model
model.fit(data)

# Predict cluster assignments
clusters, ll = model.predict(data)

# Evaluate density or likelihood
log_probs = model.ln_pdf(data)
```

---

## 🧪 API Overview

### `MultiSineBVVMMM(...)`
Initialize the mixture model.

| Parameter     | Description                                                  |
|---------------|--------------------------------------------------------------|
| `n_components`| Number of clusters                                           |
| `n_features`  | Number of bivariate angle pairs (e.g., phi/psi sites)        |
| `max_iter`    | Maximum EM iterations                                        |
| `tol`         | Convergence threshold for log-likelihood                     |
| `device`      | 'cuda' or 'cpu'                                              |
| `verbose`     | If True, prints log-likelihood during training               |

---

### 🔧 Key Methods

| Method              | Description                                          |
|---------------------|------------------------------------------------------|
| `fit(data)`         | Fit model to angular data of shape `(N, M, 2)`       |
| `predict(data)`     | Predict cluster assignments                          |
| `ln_pdf(data)`      | Log-density under the fitted model                   |
| `pdf(data)`         | Probability density under the fitted model           |
| `plot_clusters(data)` | Visualize clustering (for `M=1`)                   |

---

## 🧬 Applications

- Protein conformation analysis (Ramachandran plots)
- Directional statistics
- Robotics and cyclic motion modeling
- Meteorology and wind direction clustering
- Geospatial or angular temporal pattern discovery

---

## 📚 References

- Mardia & Jupp (2009), *Directional Statistics*
- Boomsma et al. (2008), *Bivariate von Mises for protein geometry*
- Dobson (1978), *Estimating concentration in von Mises distributions*

---

## 🙌 Acknowledgments

Developed by Martin McCullagh as part of research into high-dimensional circular clustering and protein conformational modeling.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

