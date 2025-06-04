"""
Sine Bivariate von Mises Mixture Model (BvVMM)

Provides:
- SineBVvMMM: A PyTorch-accelerated EM algorithm for fitting mixtures of sine bivariate von Mises distributions.
- fit_with_attempts: Utility for robust model fitting via multiple random initializations.
- component_scan: Grid search over different numbers of mixture components.
"""

from .core import SineBVvMMM
from .utils import fit_with_attempts, component_scan

__all__ = [
    "SineBVvMMM",
    "fit_with_attempts",
    "component_scan"
]
