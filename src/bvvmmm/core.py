import numpy as np
import torch
from torch import vmap
import matplotlib.pyplot as plt
from torch.special import i0
from scipy.special import iv, comb
from scipy.optimize import minimize
import sys
import warnings


def assert_radians(data, lower_bound=-np.pi, upper_bound=np.pi):
    # Flatten the data for a global check
    data_flat = data.flatten()
    # Check if the majority of the data falls outside the expected radian range
    if np.any(data_flat < lower_bound) or np.any(data_flat > upper_bound):
        warnings.warn("Data values appear to be outside the typical radian range "
                      f"[{lower_bound}, {upper_bound}]. Ensure that the data is provided in radians.")
        sys.exit(1)

# TorchScript-compiled function for batched log-PDF computation.
@torch.jit.script
def batched_bvm_sine_ln_pdf(phi: torch.Tensor, psi: torch.Tensor, 
                            diff_sin_prod: torch.Tensor,
                            means: torch.Tensor, kappas: torch.Tensor, 
                            norm_const: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-PDF for the bivariate von Mises distribution in a batched manner.
    
    Parameters
    ----------
    phi : Tensor of shape (n_samples, n_components)
        The φ values for each sample and component.
    psi : Tensor of shape (n_samples, n_components)
        The ψ values for each sample and component.
    diff_sin_prod : Tensor of shape (n_samples, n_components)
        Precomputed sin(phi-mu1)*sin(phi-mu2).
    means : Tensor of shape (n_components, 2)
        Mean angles for each component.
    kappas : Tensor of shape (n_components, 3)
        Concentration parameters for each component.
    norm_const : Tensor of shape (n_components,)
        Normalization constants for each component.
        
    Returns
    -------
    log_pdf : Tensor of shape (n_samples, n_components)
        The log probability density for each sample and component.
    """
    n_samples = phi.size(0)
    n_components = phi.size(1)
    # Expand component parameters to (n_samples, n_components)
    mu1 = means[:, 0].unsqueeze(0).expand(n_samples, n_components)
    mu2 = means[:, 1].unsqueeze(0).expand(n_samples, n_components)
    kappa1 = kappas[:, 0].unsqueeze(0).expand(n_samples, n_components)
    kappa2 = kappas[:, 1].unsqueeze(0).expand(n_samples, n_components)
    lam = kappas[:, 2].unsqueeze(0).expand(n_samples, n_components)
    norm_exp = norm_const.unsqueeze(0).expand(n_samples, n_components)
    
    exponent = (kappa1 * torch.cos(phi - mu1) +
                kappa2 * torch.cos(psi - mu2) +
                lam * diff_sin_prod)
    return exponent - torch.log(norm_exp)

class SineBVvMMM:
    """
    Sine Bivariate von Mises Mixture Model Expectation-Maximization (EM) Algorithm

    This class implements an Expectation-Maximization (EM) algorithm for 
    fitting a mixture model of Sine Bivariate von Mises (BVM) distributions 
    to circular data, such as protein backbone dihedral angles (phi, psi).

    Parameters
    ----------
    n_components : int, default=2
        The number of mixture components (clusters) to fit.

    small_lambda : boolean, default=True
        Boolean to dictate whether to use the small lambda approximation.  If False, numeric 
        minimization of the LL is performed after the analytic minimization in the small lambda
        approximation. 

    max_iter : int, default=100
        The maximum number of EM iterations allowed

    tol : float, default=1e-4
        The convergence tolerance for log-likelihood improvement. The EM 
        algorithm stops if the change in log-likelihood is smaller than this value.

    device : str, optional
        The computation device, either 'cpu' or 'cuda'. If not specified, 
        it defaults to 'cuda' if a GPU is available; otherwise, it falls back to 'cpu'.

    dtype : torch.dtype, default=torch.float64
        The precision type for computations. Double precision (float64) is 
        used by default to ensure numerical stability.

    seed : int, optional
        Random seed for initialization. Ensures reproducibility if set.

    verbose : bool, default=False
        If True, prints log-likelihood values and cluster weights at each iteration.

    Attributes
    ----------
    weights_ : torch.Tensor or None
        Mixture weights (probabilities of each cluster). Shape: (n_components,).

    kappas_ : torch.Tensor or None
        Concentration parameters (kappa1, kappa2) and correlation term (lambda)
        for each cluster. Shape: (n_components, 3).

    means_ : torch.Tensor or None
        Mean directions (mu1, mu2) for each cluster. Shape: (n_components, 2).

    Example
    -------
    >>> model = SineBVvMMM(n_components=3, max_iter=200, verbose=True, tol=1e-5, seed=1234)
    >>> model.fit(data)
    >>> model.plot_scatter_clusters(data)
    """

    def __init__(self, n_components=2, small_lambda=True, max_iter=100, tol=1e-4, device=None, dtype=torch.float64, seed=None, verbose=False):
        self.n_components = n_components
        self.small_lambda = small_lambda
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
        self.verbose = verbose
        self.weights_ = None
        self.kappas_ = None
        self.means_ = None

    def _calculate_normalization_constant_numpy(self, kappa1, kappa2, lam, thresh=1e-10, m_max=100):
        """
        Calculate the normalization constant C for the sine BVM using NumPy.

        Parameters
        ----------
        kappa1 : float
            Concentration parameter for phi.
        kappa2 : float
            Concentration parameter for psi.
        lam : float
            Correlation term.
        thresh : float
            Convergence threshold for infinite sum.
        m_max : int
            Maximum number of terms.

        Returns
        -------
        C : float
            Normalization constant.
        """
        C = 0.0
        diff = 1.0
        m = 0
        const = 4 * np.pi**2
        arg = lam**2 / (4 * kappa1 * kappa2)

        while diff > thresh and m < m_max:
            comb_term = comb(2*m, m)
            diff = const * comb_term * (arg ** m) * iv(m, kappa1) * iv(m, kappa2)
            C += diff
            m += 1

        return C



    def _calculate_normalization_constant(self, kappas, thresh=1e-10, m_max=100):
        """
        Calculate the normalization constant of the Sine BVM for a batch of components
        using an infinite sum that is terminated once the maximum term is below a threshold.
    
        Parameters
        ----------
        kappas : torch.Tensor, shape (n_components, 3)
            A tensor containing the concentration parameters for each component, where:
                kappas[:,0] : kappa1
                kappas[:,1] : kappa2
                kappas[:,2] : lambda (correlation term)
        thresh : float, optional
            Convergence threshold for the series (default: 1e-10).
        m_max : int, optional
            Maximum number of terms to sum (default: 100).
    
        Returns
        -------
        C : torch.Tensor, shape (n_components,)
            The normalization constant for each component.
        """
        # Move to CPU and convert to NumPy for SciPy functions.
        kappas_np = kappas.detach().cpu().numpy()
        k1 = kappas_np[:, 0]
        k2 = kappas_np[:, 1]
        lam = kappas_np[:, 2]
        #const = 4 * np.pi**2
        # Precompute the common term in the series.
        arg = lam**2 / (4 * k1 * k2)
    
        # Initialize normalization constant for each component.
        C = np.zeros_like(k1)
    
        # Sum terms from m = 0 up to m_max (or until convergence)
        for m in range(m_max):
            # Compute the mth term for all components.
            term = comb(2 * m, m) * (arg**m) * iv(m, k1) * iv(m, k2)
            C += term
            # Check convergence: if the maximum absolute term is below the threshold, break.
            if np.all(np.abs(term) < thresh):
                break
        C *= 4 * np.pi**2
        # Convert back to a torch tensor on the original device and dtype.
        return torch.tensor(C, device=self.device, dtype=self.dtype)


    def _second_order_normalization_constant(self, kappas):
        """
        Calculate the normalization constant of the Sine BVM using a second-order 
        approximation (m = 0 and m = 1 terms only) for a batched input of kappas.

        This approximation assumes that the correlation term (lambda, i.e., kappas[:,2]) 
        is small. It is fully vectorized and works on GPU or CPU.

        Parameters
        ----------
        kappas : torch.Tensor, shape (n_components, 3)
            Tensor containing the concentration parameters for each component, where:
                kappas[:,0] : kappa1
                kappas[:,1] : kappa2
                kappas[:,2] : lambda

        Returns
        -------
        C : torch.Tensor, shape (n_components,)
            The approximate normalization constant for each component.
        """
        # Compute the argument for the second-order term.
        arg = kappas[:, 2] ** 2 / (4 * kappas[:, 0] * kappas[:, 1])
    
        # Compute modified Bessel functions for each component.
        i0_k1 = torch.special.i0(kappas[:, 0])
        i0_k2 = torch.special.i0(kappas[:, 1])
        i1_k1 = torch.special.i1(kappas[:, 0])
        i1_k2 = torch.special.i1(kappas[:, 1])
    
        # Second-order approximation: m = 0 term + 2 * (m = 1 term)
        C = i0_k1 * i0_k2 + 2 * arg * i1_k1 * i1_k2
        C *= 4 * torch.pi**2
        return C
    
    def _bvm_sine_pdf(self, phi, psi, means, kappas, C=None):
        if C is None:
            C = self._calculate_normalization_constant(kappas)
        exponent = (kappas[0] * torch.cos(phi - means[0]) +
                    kappas[1] * torch.cos(psi - means[1]) +
                    kappas[2] * torch.sin(phi - means[0]) * torch.sin(psi - means[1]))
        return torch.exp(exponent) / C

    def _bvm_sine_ln_pdf(self, phi, psi, means, kappas, C):
        # This function is superseded by the batched version compiled with TorchScript.
        exponent = (kappas[0] * torch.cos(phi - means[0]) +
                    kappas[1] * torch.cos(psi - means[1]) +
                    kappas[2] * torch.sin(phi - means[0]) * torch.sin(psi - means[1]))
        return exponent - torch.log(C)
        

    def _nll_single_cluster_numpy(self, params, data, resp):
        """
        Negative log-likelihood for a single cluster using NumPy.

        Parameters
        ----------
        params : ndarray, shape (5,)
            [mu1, mu2, log(kappa1), log(kappa2), lambda].
        data : ndarray, shape (n_samples, 2)
            (phi, psi) data points in radians.
        resp : ndarray, shape (n_samples,)
            Responsibilities for the cluster.

        Returns
        -------
        loss : float
            Negative log-likelihood.
        """
        # Unpack and reparameterize
        mu1, mu2, log_kappa1, log_kappa2, eta = params
        kappa1 = np.exp(log_kappa1)
        kappa2 = np.exp(log_kappa2)
        lam = np.sqrt(kappa1*kappa2)*np.tanh(eta)

        phi, psi = data[:, 0], data[:, 1]

        # Exponent
        exponent = (
            kappa1 * np.cos(phi - mu1) +
            kappa2 * np.cos(psi - mu2) +
            lam * np.sin(phi - mu1) * np.sin(psi - mu2)
        )

        # Normalization constant
        C = self._calculate_normalization_constant_numpy(kappa1, kappa2, lam)

        # Log-likelihood per point
        log_pdf = exponent - np.log(C)

        # Weighted sum
        nll = -np.sum(resp * log_pdf)

        return nll

    def _optimize_cluster_scipy(self, data, resp, initial_guess, max_iter=100):
        """
        Minimize NLL for a single cluster using SciPy minimize.

        Parameters
        ----------
        data : ndarray, shape (n_samples, 2)
            Input data (phi, psi).
        resp : ndarray, shape (n_samples,)
            Responsibilities for the cluster.
        initial_guess : list or array
            [mu1, mu2, kappa1, kappa2, lambda] starting values.

        Returns
        -------
        optimized_params : ndarray
            Optimized [mu1, mu2, kappa1, kappa2, lambda].
        """

        # Initial guess with log transform for kappas
        mu1_init, mu2_init, kappa1_init, kappa2_init, lambda_init = initial_guess
        x0 = np.array([mu1_init, mu2_init, np.log(kappa1_init + 1e-6), np.log(kappa2_init + 1e-6), np.arctanh(lambda_init/np.sqrt(kappa1_init*kappa2_init))])

        # Bounds:
        bounds = [
            (-np.pi, np.pi),     # mu1
            (-np.pi, np.pi),     # mu2
            (None, None),        # log(kappa1) - unbounded
            (None, None),        # log(kappa2) - unbounded
            (None, None)         # eta - unbounded
        ]

        result = minimize(
            self._nll_single_cluster_numpy,
            x0,
            args=(data, resp),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )

        # Extract optimized parameters
        mu1_opt, mu2_opt, log_kappa1_opt, log_kappa2_opt, eta_opt = result.x
        kappa1_opt = np.exp(log_kappa1_opt)
        kappa2_opt = np.exp(log_kappa2_opt)
        lambda_opt = np.sqrt(kappa1_opt*kappa2_opt)*np.tanh(eta_opt)
        return torch.tensor([mu1_opt, mu2_opt, kappa1_opt, kappa2_opt, lambda_opt],dtype=self.dtype, device=self.device)

        
    def _e_step(self, data, diff_sin_prod=None):
        """
        Vectorized E-step that computes the log responsibilities and overall log-likelihood.
        Uses the TorchScript-compiled batched_bvm_sine_ln_pdf.
        """
        if diff_sin_prod is None:
            # compute diff_sin_prod
            diff_phi = torch.sin(data[:, 0].unsqueeze(1) - self.means_[:,0].unsqueeze(0))
            diff_psi = torch.sin(data[:, 1].unsqueeze(1) - self.means_[:,1].unsqueeze(0))
            diff_sin_prod = diff_phi * diff_psi
        n_samples = data.size(0)
        # data is shape (n_samples, 2)
        # Expand phi and psi to shape (n_samples, n_components)
        phi = data[:, 0].unsqueeze(1).expand(n_samples, self.n_components)
        psi = data[:, 1].unsqueeze(1).expand(n_samples, self.n_components)
        # Compute log PDF for each sample and component in one vectorized call.
        log_pdf = batched_bvm_sine_ln_pdf(phi, psi, diff_sin_prod, self.means_, self.kappas_, self.normalization_)
        # Add the log mixture weights.
        log_weights = torch.log(self.weights_).unsqueeze(0)  # shape (1, n_components)
        log_responsibilities = log_weights + log_pdf  # shape (n_samples, n_components)
        # Compute normalization over components.
        log_norm = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
        ll = torch.mean(log_norm)
        responsibilities = torch.exp(log_responsibilities - log_norm)
        return responsibilities, ll

    def _m_step(self, data, sin_data, cos_data, responsibilities):
        """
        Vectorized M-step using torch.einsum for a single pair of dihedral angles.

        Parameters
        ----------
        data : torch.Tensor, shape (n_samples, 2)
            Input data containing the dihedral angles (φ, ψ).
        sin_data : torch.Tensor, shape (n_samples, 2)
            Sine of input data containing  sin(φ, ψ).
        cos_data : torch.Tensor, shape (n_samples, 2)
            Cosine input data containing cos(φ, ψ).
        responsibilities : torch.Tensor, shape (n_samples, n_components)
            Responsibilities from the E-step.

        Returns
        -------
        weights : torch.Tensor, shape (n_components,)
            Updated mixture weights.
        means : torch.Tensor, shape (n_components, 2)
            Updated mean angles (μ_φ, μ_ψ) for each component.
        kappas : torch.Tensor, shape (n_components, 3)
            Updated concentration parameters (κ_φ, κ_ψ) and the correlation term (λ)
            for each component.
        """
        n_samples = data.shape[0]
        # Update weights: average responsibility for each component.
        weights = responsibilities.sum(dim=0) / n_samples

        # Normalize responsibilities per component: shape (n_samples, n_components)
        norm_resp = responsibilities / responsibilities.sum(dim=0, keepdim=True)

        # Compute weighted sums for φ (column 0) for each component using einsum.
        S_phi = torch.einsum('nk,n->k', norm_resp, sin_data[:, 0])
        C_phi = torch.einsum('nk,n->k', norm_resp, cos_data[:, 0])
        # Similarly for ψ (column 1).
        S_psi = torch.einsum('nk,n->k', norm_resp, sin_data[:, 1])
        C_psi = torch.einsum('nk,n->k', norm_resp, cos_data[:, 1])

        # Compute mean angles (μ) for each component.
        mu_phi = torch.atan2(S_phi, C_phi)  # shape (n_components,)
        mu_psi = torch.atan2(S_psi, C_psi)    # shape (n_components,)

        # Compute resultant lengths (R) for φ and ψ.
        R_phi = torch.sqrt(S_phi**2 + C_phi**2)
        R_psi = torch.sqrt(S_psi**2 + C_psi**2)

        # Estimate concentration parameters (κ) using the f4 approximation.
        kappa_phi = (1.28 - 0.53 * R_phi**2) * torch.tan(0.5 * torch.pi * R_phi)
        kappa_psi = (1.28 - 0.53 * R_psi**2) * torch.tan(0.5 * torch.pi * R_psi)

        # Compute Bessel function ratios for κ.
        bessel_ratio_phi = torch.special.i1(kappa_phi) / torch.special.i0(kappa_phi)
        bessel_ratio_psi = torch.special.i1(kappa_psi) / torch.special.i0(kappa_psi)

        # Compute the λ (correlation term) for each component.
        # Compute differences between each data point and the component means.
        # data[:,0] has shape (n_samples,) and mu_phi has shape (n_components,)
        # Use unsqueeze to broadcast to (n_samples, n_components).
        diff_phi = torch.sin(data[:, 0].unsqueeze(1) - mu_phi.unsqueeze(0))
        diff_psi = torch.sin(data[:, 1].unsqueeze(1) - mu_psi.unsqueeze(0))
        diff_sin_prod = diff_phi * diff_psi
        # Weighted sum over samples for each component.
        weighted_sum = torch.einsum('nk,nk->k', norm_resp, diff_sin_prod)
        lambda_val = (kappa_phi * kappa_psi * weighted_sum) / (bessel_ratio_phi * bessel_ratio_psi)

        # Assemble the updated means and kappas.
        means = torch.stack([mu_phi, mu_psi], dim=1)         # shape (n_components, 2)
        kappas = torch.stack([kappa_phi, kappa_psi, lambda_val], dim=1)  # shape (n_components, 3)
 
        # precompute normalization constants
        norms = self._calculate_normalization_constant(kappas)
        #norms = torch.tensor([self._calculate_normalization_constant(kappas[k]) for k in range(self.n_components)])
        return weights, means, kappas, norms, diff_sin_prod

    def _m_step_numeric(self, data, responsibilities, initial_guess):
        """
        Numeric Minimizsation M-step for a single pair of dihedral angles.

        Parameters
        ----------
        data : torch.Tensor, shape (n_samples, 2)
            Input data containing the dihedral angles (φ, ψ).
        responsibilities : torch.Tensor, shape (n_samples, n_components)
            Responsibilities from the E-step.
        initial_guess : numpy array, shape (n_components,5)
           Initial guess for parameters (μ_φ, μ_ψ, κ_φ, κ_ψ, λ) of each component

        Returns
        -------
        weights : torch.Tensor, shape (n_components,)
            Updated mixture weights.
        means : torch.Tensor, shape (n_components, 2)
            Updated mean angles (μ_φ, μ_ψ) for each component.
        kappas : torch.Tensor, shape (n_components, 3)
            Updated concentration parameters (κ_φ, κ_ψ) and the correlation term (λ)
            for each component.
        """
        n_samples = data.shape[0]
        # Update weights: average responsibility for each component.
        weights = responsibilities.sum(dim=0) / n_samples

        # Normalize responsibilities per component: shape (n_samples, n_components)
        norm_resp = responsibilities / responsibilities.sum(dim=0, keepdim=True)

        # declare tensors
        means = torch.empty((self.n_components,2),dtype=self.dtype, device=self.device)
        kappas = torch.empty((self.n_components,3),dtype=self.dtype, device=self.device)
    
        # loop over components and compute optimal parameters using numeric minimization of LL
        for k in range(self.n_components):
            # Optimize with scipy
            optimized_params = self._optimize_cluster_scipy(data.cpu().numpy(), norm_resp[:, k].cpu().numpy(), initial_guess[k])
            # Assign back
            means[k, 0], means[k, 1] = optimized_params[0], optimized_params[1]
            kappas[k, 0], kappas[k, 1], kappas[k, 2] = optimized_params[2], optimized_params[3], optimized_params[4]
        # precompute diff_sin_prod
        diff_phi = torch.sin(data[:, 0].unsqueeze(1) - means[:,0].unsqueeze(0))
        diff_psi = torch.sin(data[:, 1].unsqueeze(1) - means[:,1].unsqueeze(0))
        diff_sin_prod = diff_phi * diff_psi
        # precompute normalization constants
        norms = self._calculate_normalization_constant(kappas)
        return weights, means, kappas, norms, diff_sin_prod
    
    def fit(self, data):
        """
        Fit the Sine Bivariate von Mises Mixture Model (SineBVvMMM) to the given data using the 
        Expectation-Maximization (EM) algorithm.

        This method performs two phases of optimization:
        (1) **Analytic Initialization Phase** (Small lambda approximation):
            - Updates cluster parameters using closed-form moment estimators 
            - Fast and guarantees initial unimodal solutions.

        (2) **Numeric Refinement Phase** (Optional, if `small_lambda=False`):
            - Refines the cluster parameters by directly minimizing the negative 
              log-likelihood via numeric optimization (SciPy `L-BFGS-B`).
            - Ensures high-accuracy parameter estimates without small lambda assumptions.
            - Enforces unimodality through reparameterization of lambda via `tanh`.

        Parameters
        ----------
        data : ndarray, shape (n_samples, 2)
            Input data matrix where each row is a (phi, psi) dihedral angle pair in radians.
            Values must be in the range [-π, π).

        Notes
        -----
        - The data is internally converted to a PyTorch tensor with specified `device` and `dtype`.
        - The method checks that data is in radians and warns if it is outside the expected range.
        - Log-likelihood convergence is monitored based on the tolerance `tol`.
        - The method records the final log-likelihood in `self.ll` and sorts clusters by weight after fitting.

        Warnings
        --------
        A `UserWarning` is issued if the EM algorithm does not converge within `max_iter` iterations.

        See Also
        --------
        predict : Predict the most likely cluster labels for new data.
        ln_pdf : Compute the log-probability density function (log-pdf) for input data.
        pdf : Compute the probability density function (pdf) for input data.
        """

        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        # Precompute sin and cos values
        sin_data = torch.sin(data)
        cos_data = torch.cos(data)
        # initialize Model parameters
        self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, device=self.device, dtype=self.dtype)
        self.means_ = torch.rand((self.n_components,2), device=self.device, dtype=self.dtype) * 2 * torch.pi - torch.pi
        self.kappas_ = torch.column_stack( (torch.ones((self.n_components,2), device=self.device, dtype=self.dtype),torch.zeros(self.n_components, device=self.device, dtype=self.dtype)))
        self.normalization_ = self._second_order_normalization_constant(self.kappas_)
        # precompute diff_sin_prod
        diff_phi = torch.sin(data[:, 0].unsqueeze(1) - self.means_[:,0].unsqueeze(0))
        diff_psi = torch.sin(data[:, 1].unsqueeze(1) - self.means_[:,1].unsqueeze(0))
        diff_sin_prod = diff_phi * diff_psi
        # perform EM:
        old_ll = 0.0
        with torch.no_grad():  # Disable gradients
            for _ in range(self.max_iter):
                # e-step
                responsibilities, ll = self._e_step(data, diff_sin_prod)
                # m-step
                self.weights_, self.means_, self.kappas_, self.normalization_, diff_sin_prod  = self._m_step(data, sin_data, cos_data, responsibilities)
    
                if self.verbose:
                    print(_, ll.cpu().numpy(), self.weights_.cpu().numpy())
            
                if torch.abs(ll - old_ll) < self.tol:
                    break
                old_ll = ll
            # refine with numeric minimization if requested (small_lambda == False)
            if self.small_lambda == False:
                for _ in range(self.max_iter):
                    # Create initial guess array
                    initial_guess_tensor = torch.cat([self.means_, self.kappas_], dim=1)  # shape (n_components, 5)
                    initial_guess = initial_guess_tensor.cpu().numpy()
                    self.weights_, self.means_, self.kappas_, self.normalization_, diff_sin_prod  = self._m_step_numeric(data, responsibilities, initial_guess)
                    # e step
                    responsibilities, ll = self._e_step(data, diff_sin_prod)
                    if self.verbose:
                        print(_, ll.cpu().numpy(), self.weights_.cpu().numpy())
            
                    if torch.abs(ll - old_ll) < self.tol:
                        break
                    old_ll = ll
                    
        self.ll = ll
        # sort by cluster weights
        self.sort_clusters()

    def predict(self, data):
        """
        Assign the most likely cluster to each sample in the input data.

        This method computes the posterior responsibilities for each data point and 
        returns the cluster with the highest responsibility (i.e., maximum a posteriori estimate).

        Parameters
        ----------
        data : ndarray, shape (n_samples, 2)
            Input data matrix where each row is a (phi, psi) dihedral angle pair in radians.
            Values must be within the range [-π, π).

        Returns
        -------
        labels : torch.Tensor, shape (n_samples,)
            The predicted cluster labels for each sample, as integer indices in [0, n_components - 1].

        ll : torch.Tensor
            The average log-likelihood of the input data under the fitted mixture model.

        Notes
        -----
        - The input data is validated to ensure it is in radians and converted to a PyTorch tensor 
          with the appropriate device and dtype.
        - The cluster labels are determined by maximum posterior probability per sample.

        See Also
        --------
        fit : Fit the model to data.
        ln_pdf : Compute the log-probability density function (log-pdf) for input data.
        pdf : Compute the probability density function (pdf) for input data.
        """
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data, device=self.device, dtype=self.dtype)
        responsibilities, ll = self._e_step(data)
        return responsibilities.argmax(dim=1), ll

    def sort_clusters(self):
        """
        Sort clusters by descending cluster weight. All corresponding parameters (means, kappas, normalization constants)
        are rearranged accordingly.
        """
        # Get the sorted indices (descending order)
        sorted_indices = torch.argsort(self.weights_, descending=True)
        # Rearrange all cluster-level parameters using the sorted indices.
        self.weights_ = self.weights_[sorted_indices]
        self.means_ = self.means_[sorted_indices]
        self.kappas_ = self.kappas_[sorted_indices]
        if self.normalization_ is not None:
            self.normalization_ = self.normalization_[sorted_indices]

    def ln_pdf(self, data):
        """ Compute ln pdf of the mixture for points """
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data, device=self.device, dtype=self.dtype)
        # precompute diff_sin_prod
        diff_phi = torch.sin(data[:, 0].unsqueeze(1) - self.means_[:,0].unsqueeze(0))
        diff_psi = torch.sin(data[:, 1].unsqueeze(1) - self.means_[:,1].unsqueeze(0))
        diff_sin_prod = diff_phi * diff_psi
        # Vectorized log-PDF evaluation.
        n_samples = data.size(0)
        phi = data[:, 0].unsqueeze(1).expand(n_samples, self.n_components)
        psi = data[:, 1].unsqueeze(1).expand(n_samples, self.n_components)
        log_pdf = batched_bvm_sine_ln_pdf(phi, psi, diff_sin_prod, self.means_, self.kappas_, self.normalization_)
        log_weights = torch.log(self.weights_).unsqueeze(0)
        ln_likelihoods = log_pdf + log_weights
        return torch.logsumexp(ln_likelihoods, dim=1)

    def pdf(self, data):
        """ Compute pdf of the mixture for points """
        # check data is in radians
        assert_radians(data)
        #
        return torch.exp(self.ln_pdf(data)).cpu().numpy()

    def icl(self, data):
        """Integrated completed likelihood per frame (McNicholas eq 4)"""
        n_samples = data.shape[0]
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        # precompute diff_sin_prod
        diff_phi = torch.sin(data[:, 0].unsqueeze(1) - self.means_[:,0].unsqueeze(0))
        diff_psi = torch.sin(data[:, 1].unsqueeze(1) - self.means_[:,1].unsqueeze(0))
        diff_sin_prod = diff_phi * diff_psi
        # Expand phi and psi to shape (n_samples, n_components)
        phi = data[:, 0].unsqueeze(1).expand(n_samples, self.n_components)
        psi = data[:, 1].unsqueeze(1).expand(n_samples, self.n_components)
        log_pdf = batched_bvm_sine_ln_pdf(phi, psi, diff_sin_prod, self.means_, self.kappas_, self.normalization_)
        # Add the log mixture weights.
        log_weights = torch.log(self.weights_).unsqueeze(0)  # shape (1, n_components)
        log_responsibilities = log_weights + log_pdf  # shape (n_samples, n_components)
        # Compute normalization over components.
        log_norm = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
        ll = torch.mean(log_norm)
        temp = torch.mean(torch.amax(log_responsibilities,axis=1)).cpu().numpy()
        return self.n_components*6*np.log(n_samples)/n_samples - 2*ll - 2*temp

    def bic(self, data):
        """Bayesian Information Criterion per frame"""
        n_frames = data.shape[0]
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return self.n_components*6*np.log(n_frames)/n_frames - 2*ll 

    def aic(self, data):
        """Akaike Information Criterion per frame"""
        n_frames = data.shape[0]
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return self.n_components*12/n_frames - 2*ll 
    
    def plot_scatter_clusters(self, data, title=None):
        fontsize=12
        clusters = self.predict(data)[0].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='tab10', s=10)
        plt.xlabel(r'$\phi$ (radians)',fontsize=fontsize)
        plt.ylabel(r'$\psi$ (radians)',fontsize=fontsize)
        if title is None:
            title = "BVvM Mixture Model with " + str(self.n_components) + " components"
        plt.title(title, fontsize=fontsize)
        plt.grid(True)
        plt.tick_params(labelsize=fontsize)
        plt.xlim(-np.pi,np.pi)
        plt.ylim(-np.pi,np.pi)
        plt.axis('equal')
        plt.tight_layout()
        plt.show();
        
    def plot_model_sample_fe(self, data, vmin=0, vmax=10, title=None, axes=None, filename=None, fontsize=12):
        #make sure data is provided in radians - quit if not
        assert_radians(data)
        # ignore divide by zero error message from numpy
        np.seterr(divide = 'ignore') 
        # Create the figure and subplots
        show_plot = False
        if axes is None:
            show_plot = True
            fig, axes = plt.subplots(1, 1, figsize=(5, 5)) # 1 row, 1 column
        # set some grid stuff
        theta = np.linspace(-np.pi, np.pi, 200)
        phi_mesh, psi_mesh = np.meshgrid(theta,theta)
        phi_psi_grid = np.column_stack((phi_mesh.flatten(),psi_mesh.flatten()))
    
        # determine sample  FE
        hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=np.linspace(-np.pi,np.pi,200), density=True)
        x = 0.5*(xedges[1:] + xedges[:-1])
        y = 0.5*(yedges[1:] + yedges[:-1])
        Y, X = np.meshgrid(x,y)
        sample_fe = -np.log(hist)
        sample_fe -= np.amin(sample_fe)
        
        # determine model marginal FE
        model_fe = -self.ln_pdf(phi_psi_grid).cpu().numpy()
        model_fe -= np.amin(model_fe)
        model_fe = model_fe.reshape(phi_mesh.shape)
        # plot
        if title is None:
            title =  "FE/kT (" + str(self.n_components) + " components)"
 
        axes.pcolormesh(phi_mesh, psi_mesh, model_fe, cmap='hot_r', vmin=0, vmax=vmax)
        axes.contour(X, Y, sample_fe,alpha=0.5, vmin=vmin, vmax=vmax)
        # scatter means
        axes.scatter(self.means_[:,0], self.means_[:,1], s=80*self.weights_)
        axes.set_xlabel(r'$\phi$ (radians)', fontsize=fontsize)
        axes.set_ylabel(r'$\psi$ (radians)', fontsize=fontsize)
        axes.set_title(title, fontsize=fontsize)
        axes.tick_params(labelsize=fontsize)

        # savefig if desired
        if filename is not None:
            plt.savefig(filename,dpi=80)
        # show plot if desired
        if show_plot == True:
            #plot params
            plt.tight_layout()
            # Show the plot
            plt.show();        
