import torch
from functorch import vmap
import matplotlib.pyplot as plt
from torch.special import i0
from scipy.special import iv, comb
import numpy as np
import warnings
import sys

def fit_with_attempts(data, n_components, n_attempts):
    models = []
    ll = np.empty(n_attempts)
    for i in range(n_attempts):
        model = MultiSineBVVMMM(n_components=n_components, max_iter=200, tol=1e-5)
        model.fit(data)
        models.append(model)
        ll[i] = model.ll
        print(i+1, ll[i])
    return models[np.nanargmax(ll)]

def component_scan(data, components, n_attempts=15, tol=1e-4, train_frac=1.0, verbose=True, plot=False):
    """
    Scan through different numbers of components by fitting multiple attempts
    of the SineVMEM model and returning metrics including training log likelihood,
    AIC, BIC, ICL, and (if using a validation split) the cross validation log likelihood.

    Parameters
    ----------
    data : array-like, shape (n_data_points, n_residues, 2)
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

    for i, comp in enumerate(components):
        temp_ll = np.empty(n_attempts)
        if cv_data is not None:
            temp_cv_ll = np.empty(n_attempts)
        models = []
        for attempt in range(n_attempts):
            model = MultiSineBVVMMM(n_components=comp, max_iter=200, verbose=False, tol=tol)
            # Fit using the training set only
            model.fit(train_data)
            models.append(model)
            temp_ll[attempt] = model.ll.cpu().numpy()
            if cv_data is not None:
                # Evaluate CV log likelihood on the held-out set.
                # Note: model._e_step returns (responsibilities, log_likelihood)
                _, cv_loglik = model._e_step(torch.tensor(cv_data, device=model.device, dtype=model.dtype))
                temp_cv_ll[attempt] = cv_loglik.cpu().numpy()
            if verbose == True:
                print(f"Components: {comp}, Attempt: {attempt+1}, Training LL: {temp_ll[attempt]}, " +
                  (f"CV LL: {temp_cv_ll[attempt]}" if cv_data is not None else ""))
        # Choose the attempt with the highest training log likelihood
        best_index = np.nanargmax(temp_ll)
        ll[i] = temp_ll[best_index]
        best_model = models[best_index]
        # plot
        if plot==True:
            best_model.plot_model_sample_residue_marginal_fe(data)
        # Compute the information criteria on the full data (or you could use train_data if preferred)
        aic[i] = best_model.aic(data)
        bic[i] = best_model.bic(data)
        icl[i] = best_model.icl(data)
        if cv_data is not None:
            cv_ll[i] = temp_cv_ll[best_index]

    if cv_data is not None:
        return ll, aic, bic, icl, cv_ll
    else:
        return ll, aic, bic, icl


def assert_radians(data, lower_bound=-np.pi, upper_bound=np.pi):
    # Flatten the data for a global check
    data_flat = data.flatten()
    # Check if the majority of the data falls outside the expected radian range
    if np.any(data_flat < lower_bound) or np.any(data_flat > upper_bound):
        warnings.warn("Data values appear to be outside the typical radian range "
                      f"[{lower_bound}, {upper_bound}]. Ensure that the data is provided in radians.")
        sys.exit(1)

class MultiSineBVVMMM:
    """
    Sine Bivariate von Mises Mixture Model Expectation-Maximization (EM) Algorithm

    This class implements an Expectation-Maximization (EM) algorithm for 
    fitting a mixture model of Sine Bivariate von Mises (BVM) distributions 
    to circular data, such as protein backbone dihedral angles (phi, psi).

    Parameters
    ----------
    n_components : int, default=2
        The number of mixture components (clusters) to fit.

    max_iter : int, default=100
        The maximum number of EM iterations before convergence.

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
    >>> model = MultiSineBVVMMM(n_components=3, max_iter=200, verbose=True, tol=1e-5, seed=1234)
    >>> model.fit(data)
    >>> model.plot_clusters(data)
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-4, device=None, dtype=torch.float64, seed=None, verbose=False):
        self.n_components = n_components
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

    def _calculate_normalization_constant(self, kappas, thresh=1e-10, m_max=100):
        """ Calculate the normalization constant of the Sine BVM using (hopefully) converging infinite sum 
            NOTE: This is currently done on cpu-only using numpy because of the need of iv function
        """
        k1 = kappas[0].cpu().numpy()
        k2 = kappas[1].cpu().numpy()
        lam = kappas[2].cpu().numpy()
        C = 0
        diff = 1
        m = 0
        const = 4*np.pi**2
        arg = lam ** 2 / (4 * k1 * k2)
        while diff > thresh and m < m_max:
            diff = const * comb(2 * m, m) * arg ** m * iv(m,k1) * iv(m,k2)
            C += diff
            m += 1
        return torch.tensor(C, device=self.device, dtype=self.dtype)

    def _bvm_sine_pdf(self, phi, psi, means, kappas):
        C = self._calculate_normalization_constant(kappas)
        exponent = (kappas[0] * torch.cos(phi - means[0]) +
                    kappas[1] * torch.cos(psi - means[1]) +
                    kappas[2] * torch.sin(phi - means[0]) * torch.sin(psi - means[1]))
        return torch.exp(exponent) / C

    def _bvm_sine_ln_pdf(self, phi, psi, means, kappas):
        C = self._calculate_normalization_constant(kappas)
        exponent = (kappas[0] * torch.cos(phi - means[0]) +
                    kappas[1] * torch.cos(psi - means[1]) +
                    kappas[2] * torch.sin(phi - means[0]) * torch.sin(psi - means[1]))
        return exponent - torch.log(C)
        
    def _independent_bvm_sine_ln_pdf(self, phis, psis, means, kappas):

        ll = 0.0
        for i in range(self.n_residues):
            ll += self._bvm_sine_ln_pdf(phis[:,i], psis[:,i], means[i], kappas[i])
        return ll
    
    def _e_step(self, data):
        # compute log responsibilities
        log_responsibilities = torch.stack([
            torch.log(self.weights_[k]) + self._independent_bvm_sine_ln_pdf(data[:, :, 0], data[:, :, 1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        # log norm for each data point
        log_norm = torch.logsumexp(log_responsibilities,dim=1,keepdim=True)
        # log likelihood per data point
        ll = torch.mean(log_norm)
        # compute responsibilities
        responsibilities = torch.exp(log_responsibilities - log_norm)
        return responsibilities, ll

    def _m_step_vectorized(self, data, sin_data, cos_data, responsibilities):
        """
        Vectorized version of the M-step.
    
        Parameters
        ----------
        data : torch.Tensor, shape (n_samples, n_residues, 2)
            Input data, where the last dimension holds (phi, psi).
        sin_data : torch.Tensor, shape (n_samples, n_residues, 2)
            Sine of input data, where the last dimension holds (sin(phi), sin(psi)).
        cos_data : torch.Tensor, shape (n_samples, n_residues, 2)
            Sine of input data, where the last dimension holds (cos(phi), cos(psi)).
        responsibilities : torch.Tensor, shape (n_samples, n_components)
            The responsibilities computed in the E-step.
    
        Returns
        -------
        weights : torch.Tensor, shape (n_components,)
            Updated mixture weights.
        means : torch.Tensor, shape (n_components, n_residues, 2)
            Updated mean directions for each cluster and residue.
        kappas : torch.Tensor, shape (n_components, n_residues, 3)
            Updated concentration parameters (kappa1, kappa2) and lambda (correlation).
        """
        n_samples, n_residues, _ = data.shape
        n_components = self.n_components

        # Update weights: average responsibility for each component.
        weights = responsibilities.sum(dim=0) / n_samples

        # Normalize responsibilities for each component (shape: [n_samples, n_components])
        norm_resp = responsibilities / responsibilities.sum(dim=0, keepdim=True)

        # Extract phi and psi from data: shape (n_samples, n_residues)
        data_phi = data[:, :, 0]
        data_psi = data[:, :, 1]

        # Extract sine and cosine values: shape (n_samples, n_residues)
        sin_phi = sin_data[:, :, 0]
        cos_phi = cos_data[:, :, 0]
        sin_psi = sin_data[:, :, 1]
        cos_psi = cos_data[:, :, 1]

        # Compute weighted sums for phi using einsum:
        # S_bar_phi: (n_components, n_residues)
        S_bar_phi = torch.einsum('nk,nm->km', norm_resp, sin_phi)
        C_bar_phi = torch.einsum('nk,nm->km', norm_resp, cos_phi)
        mu1 = torch.atan2(S_bar_phi, C_bar_phi)  # Mean phi for each (component, residue)

        # For psi:
        S_bar_psi = torch.einsum('nk,nm->km', norm_resp, sin_psi)
        C_bar_psi = torch.einsum('nk,nm->km', norm_resp, cos_psi)
        mu2 = torch.atan2(S_bar_psi, C_bar_psi)  # Mean psi for each (component, residue)

        # Compute resultant lengths
        R1 = torch.sqrt(S_bar_phi**2 + C_bar_phi**2)  # (n_components, n_residues)
        R2 = torch.sqrt(S_bar_psi**2 + C_bar_psi**2)    # (n_components, n_residues)

        # Estimate kappas using the f4 approximation (Dobson 1978)
        kappa1 = (1.28 - 0.53 * R1**2) * torch.tan(0.5 * torch.pi * R1)
        kappa2 = (1.28 - 0.53 * R2**2) * torch.tan(0.5 * torch.pi * R2)

        # Compute Bessel function ratios for each (component, residue)
        bessel_ratio_k1 = torch.special.i1(kappa1) / torch.special.i0(kappa1)
        bessel_ratio_k2 = torch.special.i1(kappa2) / torch.special.i0(kappa2)

        # To compute lambda, we need the weighted sum of sin differences.
        # Expand mu1 and mu2 to subtract from each data point.
        # mu1 and mu2: (n_components, n_residues) -> (1, n_components, n_residues)
        mu1_exp = mu1.unsqueeze(0)
        mu2_exp = mu2.unsqueeze(0)
        # Expand data to (n_samples, 1, n_residues)
        data_phi_exp = data_phi.unsqueeze(1)
        data_psi_exp = data_psi.unsqueeze(1)
        # Compute differences: (n_samples, n_components, n_residues)
        diff_phi = data_phi_exp - mu1_exp
        diff_psi = data_psi_exp - mu2_exp

        # Expand normalized responsibilities: (n_samples, n_components, 1)
        norm_resp_exp = norm_resp.unsqueeze(2)
        # Compute the weighted sum over samples for lambda numerator.
        weighted_sum = torch.sum(norm_resp_exp * torch.sin(diff_phi) * torch.sin(diff_psi), dim=0)  # (n_components, n_residues)

        # Compute lambda for each component and residue.
        lambda_val = (kappa1 * kappa2 * weighted_sum) / (bessel_ratio_k1 * bessel_ratio_k2)

        # Stack kappas into one tensor: shape (n_components, n_residues, 3)
        kappas = torch.stack([kappa1, kappa2, lambda_val], dim=2)
        # Stack means into one tensor: shape (n_components, n_residues, 2)
        means = torch.stack([mu1, mu2], dim=2)

        return weights, means, kappas

    def _m_step(self, data, sin_data, cos_data, responsibilities):
        # determine new cluster weights based on expectation step
        weights = responsibilities.sum(dim=0) / responsibilities.size(0)
        # declare empty parameter tensors
        means = torch.empty((self.n_components, self.n_residues, 2),device=self.device, dtype=self.dtype)
        kappas = torch.empty((self.n_components, self.n_residues, 3),device=self.device, dtype=self.dtype)
        # determine new values of parameters
        for k in range(self.n_components):
            # Normalize frame weights for each cluster
            resp = responsibilities[:, k]
            resp /= resp.sum()
            for m in range(self.n_residues):
                # determine ML estimate of mu1
                S_bar = (resp * sin_data[:,m,0]).sum()
                C_bar = (resp * cos_data[:,m,0]).sum()
                R1 = torch.sqrt(C_bar**2+S_bar**2) 
                means[k,m,0] = torch.atan2(S_bar, C_bar)
                # determine ML estimate of mu2
                S_bar = (resp * sin_data[:,m,1]).sum()
                C_bar = (resp * cos_data[:,m,1]).sum()
                R2 = torch.sqrt(C_bar**2+S_bar**2) 
                means[k,m,1] = torch.atan2(S_bar, C_bar)
                # determine ML estimate of kappas currently assuming lambda is small (i.e. truncate sum at m=0)
                # use f4 approximation (min error in Table 1) from Dobson 1978 to determine estimate of kappa1 and kappa2 - this assumes lambda is zero
                kappas[k,m,0] = (1.28-0.53*R1**2)*torch.tan(0.5*torch.pi*R1)
                kappas[k,m,1] = (1.28-0.53*R2**2)*torch.tan(0.5*torch.pi*R2)
                # ML estimate of lambda
                # Compute Bessel ratios
                bessel_ratio_k1 = torch.special.i1(kappas[k,m,0]) / torch.special.i0(kappas[k,m,0])
                bessel_ratio_k2 = torch.special.i1(kappas[k,m,1]) / torch.special.i0(kappas[k,m,1])
                # Weighted estimate for lambda
                kappas[k,m,2] = (kappas[k,m,0] * kappas[k,m,1] * (resp * torch.sin(data[:,m, 0] - means[k,m,0]) * torch.sin(data[:, m, 1] - means[k, m, 1])).sum()) / (bessel_ratio_k1 * bessel_ratio_k2)

        return weights, means, kappas

    def fit(self, data):
        """ Fit the model parameters using data 
            data should be of size n_data_points x n_residues x 2
        """
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        # Precompute sin and cos values
        sin_data = torch.sin(data)
        cos_data = torch.cos(data)
        # grab metadata
        self.n_residues = data.shape[1]
        self.n_data_points = data.shape[0]
        # initialize Model parameters
        self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, device=self.device, dtype=self.dtype)
        self.means_ = torch.rand((self.n_components,self.n_residues,2), device=self.device, dtype=self.dtype) * 2 * torch.pi - torch.pi
        self.kappas_ =  torch.ones((self.n_components,self.n_residues,3), device=self.device, dtype=self.dtype)
        self.kappas_[:,:,2] = 0.0
        #self.kappas_ = torch.stack( (torch.ones((self.n_components,self.n_residues,2), device=self.device, dtype=self.dtype),torch.zeros((self.n_components,self.n_residues,1), device=self.device, dtype=self.dtype)),dim=2)

        # perform EM:
        old_ll = 0.0
        with torch.no_grad():  # Disable gradients
            for _ in range(self.max_iter):
                responsibilities, ll = self._e_step(data)
                self.weights_, self.means_, self.kappas_ = self._m_step_vectorized(data, sin_data, cos_data, responsibilities)
            
                if self.verbose:
                    print(_, ll.cpu().numpy(), self.weights_.cpu().numpy())
            
                if torch.abs(ll - old_ll) < self.tol:
                    break
                old_ll = ll
        self.ll = ll


    def predict(self, data):
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        responsibilities, ll = self._e_step(data)
        return responsibilities.argmax(dim=1).cpu().numpy(), ll.cpu().numpy()
        
    def ln_pdf(self, data):
        """ Compute ln pdf of the mixture for points """
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._independent_bvm_sine_ln_pdf(data[:,:,0], data[:,:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        return torch.logsumexp(ln_likelihoods_per_cluster,1)

    def pdf(self, data):
        """ Compute pdf of the mixture for points """
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._independent_bvm_sine_ln_pdf(data[:,:,0], data[:,:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        return torch.sum(likelihoods_per_cluster,1).cpu().numpy()

    def icl(self, data):
        """Integrated completed likelihood per frame (McNicholas eq 4 and need to read citations)"""
        n_frames = data.shape[0]
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._independent_bvm_sine_ln_pdf(data[:,:,0], data[:,:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        ll = torch.mean(torch.logsumexp(ln_likelihoods_per_cluster,1)).cpu().numpy()
        temp = torch.mean(torch.amax(ln_likelihoods_per_cluster,axis=1)).cpu().numpy()
        return self.n_components*(1+5*self.n_residues)*np.log(n_frames)/n_frames - 2*ll - 2*temp

    def bic(self, data):
        """Bayesian Information Criterion per frame"""
        n_frames = data.shape[0]
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return self.n_components*(1+5*self.n_residues)*np.log(n_frames)/n_frames - 2*ll 

    def aic(self, data):
        """Akaike Information Criterion per frame"""
        n_frames = data.shape[0]
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # pass data to torch
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return 2*self.n_components*(1+5*self.n_residues)/n_frames - 2*ll 
    
    def plot_scatter_clusters(self, data):
        # make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        fontsize=12
        clusters = self.predict(data)[0]
        # make scatter plot colored by clustering
        fig, axes = plt.subplots(1,self.n_residues,figsize=(5*self.n_residues,5))
        if self.n_residues > 1:
            for i, axis in enumerate(axes):
                axis.scatter(data[:, i, 0], data[:, i, 1], c=clusters, cmap='tab10', s=10)
                axis.set_xlabel(r'$\phi$ (radians)',fontsize=fontsize)
                axis.set_ylabel(r'$\psi$ (radians)',fontsize=fontsize)
                title = "Phi/Psi " + str(i+1) + "BVM with " + str(self.n_components) + " components"
                axis.set_title(title, fontsize=fontsize)
                axis.grid(True)
                axis.tick_params(labelsize=fontsize)
                axis.set_xlim(-np.pi,np.pi)
                axis.set_ylim(-np.pi,np.pi)
                axis.set_aspect('equal', 'box')
        else:
            axes.scatter(data[:, 0, 0], data[:, 0, 1], c=clusters, cmap='tab10', s=10)
            axes.set_xlabel(r'$\phi$ (radians)',fontsize=fontsize)
            axes.set_ylabel(r'$\psi$ (radians)',fontsize=fontsize)
            title = "Phi/Psi BVM with " + str(self.n_components) + " components"
            axes.set_title(title, fontsize=fontsize)
            axes.grid(True)
            axes.tick_params(labelsize=fontsize)
            axes.set_xlim(-np.pi,np.pi)
            axes.set_ylim(-np.pi,np.pi)
            axes.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

    def plot_model_sample_residue_marginal_fe(self, data, filename=None, fontsize=12):
        #make sure data is provided in radians - quit if not
        assert_radians(np.array(data))
        # ignore divide by zero error message from numpy
        np.seterr(divide = 'ignore') 
        # Create the figure and subplots
        fig, axes = plt.subplots(1, self.n_residues, figsize=(5*self.n_residues, 5), sharey=True) # 1 row, n_residue columns
        # set some grid stuff
        theta = np.linspace(-np.pi, np.pi, 200)
        phi_mesh, psi_mesh = np.meshgrid(theta,theta)
        phi_grid = torch.tensor(phi_mesh,device=self.device, dtype=self.dtype)
        psi_grid = torch.tensor(psi_mesh,device=self.device, dtype=self.dtype)
        # loop over residues
        for i in range(self.n_residues):
        
            # determine sameple marginal FE
            hist, xedges, yedges = np.histogram2d(data[:,i,0], data[:,i,1], bins=120, density=True)
            x = 0.5*(xedges[1:] + xedges[:-1])
            y = 0.5*(yedges[1:] + yedges[:-1])
            Y, X = np.meshgrid(x,y)
            sample_fe = -np.log(hist)
            sample_fe -= np.amin(sample_fe)
        
            # determine model marginal FE
            Z = torch.zeros_like(phi_grid)
            for j in range(self.n_components):
                Z += self.weights_[j]*self._bvm_sine_pdf(phi_grid,psi_grid,self.means_[j,i],self.kappas_[j,i])
            Z = Z.cpu().numpy()
            model_fe = -np.log(Z)
            model_fe -= np.amin(model_fe)
            
            # plot
            title = "Residue " + str(i+1) + " marginal FE (" + str(self.n_components) + " components)"
            if self.n_residues > 1:
                axes[i].pcolormesh(phi_mesh, psi_mesh, model_fe, cmap='hot_r', vmin=0, vmax=8)
                axes[i].contour(X, Y, sample_fe,alpha=0.5)
                axes[i].set_xlabel(r'$\phi$ (radians)', fontsize=fontsize)
                if i==0:
                    axes[i].set_ylabel(r'$\psi$ (radians)', fontsize=fontsize)
                axes[i].set_title(title, fontsize=fontsize)
                axes[i].tick_params(labelsize=fontsize)
            else:
                axes.pcolormesh(phi_mesh, psi_mesh, model_fe, cmap='hot_r', vmin=0, vmax=8)
                axes.contour(X, Y, sample_fe,alpha=0.5)
                axes.set_xlabel(r'$\phi$ (radians)', fontsize=fontsize)
                if i==0:
                    axes.set_ylabel(r'$\psi$ (radians)', fontsize=fontsize)
                axes.set_title(title, fontsize=fontsize)
                axes.tick_params(labelsize=fontsize)
            
        
        
        # Add color bar
        #fig.colorbar(pm, label="Free Energy/kT")
        plt.tight_layout()
        # savefig if desired
        if filename is not None:
            plt.savefig(filename,dpi=80)
        # Show the plot
        plt.show();        
