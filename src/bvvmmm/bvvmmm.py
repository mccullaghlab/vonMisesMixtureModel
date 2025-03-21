import torch
from functorch import vmap
import matplotlib.pyplot as plt
from torch.special import i0
from scipy.special import iv, comb

def fit_with_attempts(data, n_components, n_attempts, tol=1e-5):
    models = []
    ll = np.empty(n_attempts)
    for i in range(n_attempts):
        model = SineBVVMMM(n_components=n_components, max_iter=200, tol=tol)
        model.fit(data)
        models.append(model)
        ll[i] = model.ll
        print(i+1, ll[i])
    return models[np.argmax(ll)]

def component_scan(data, components, n_attempts=15, tol=1e-15):
    
    ll = np.empty(components.size)
    aic = np.empty(components.size)
    bic = np.empty(components.size)
    icl = np.empty(components.size)
    
    for i, component in enumerate(components):
        temp_ll = np.empty(n_attempts)
        models = []
        for attempt in range(n_attempts):
            model = SineBVVMMM(n_components=component, max_iter=200, verbose=False, tol=tol)
            model.fit(data)
            models.append(model)
            temp_ll[attempt] = model.ll.cpu().numpy()
            print(component, attempt+1, temp_ll[attempt])
        ll[i] = np.amax(temp_ll)
        aic[i] = models[np.argmax(temp_ll)].aic(data)
        bic[i] = models[np.argmax(temp_ll)].bic(data)
        icl[i] = models[np.argmax(temp_ll)].icl(data)
    return ll, aic, bic, icl

def assert_radians(data, lower_bound=-np.pi, upper_bound=np.pi):
    # Flatten the data for a global check
    data_flat = data.flatten()
    # Check if the majority of the data falls outside the expected radian range
    if np.any(data_flat < lower_bound) or np.any(data_flat > upper_bound):
        warnings.warn("Data values appear to be outside the typical radian range "
                      f"[{lower_bound}, {upper_bound}]. Ensure that the data is provided in radians.")

class SineBVVMMM:
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
    >>> model = SineBVVMMM(n_components=3, max_iter=200, verbose=True, tol=1e-5, seed=1234)
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
                    kappas[2] * torch.sin(phi - means[0]) * torch.sin(psi - means[0]))
        return exponent - torch.log(C)
    
    def _e_step(self, data):
        phi, psi = data[:, 0], data[:, 1]
        # compute log responsibilities
        log_responsibilities = torch.stack([
            torch.log(self.weights_[k]) + self._bvm_sine_ln_pdf(phi, psi, self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        # log norm for each data point
        log_norm = torch.logsumexp(log_responsibilities,dim=1,keepdim=True)
        # log likelihood per data point
        ll = torch.mean(log_norm)
        # compute responsibilities
        responsibilities = torch.exp(log_responsibilities - log_norm)
        return responsibilities, ll

    @staticmethod
    def _compute_cluster_params(resp, phi, psi, sin_phi, cos_phi, sin_psi, cos_psi):
        resp /= resp.sum()
    
        # Compute means
        S_bar_phi = (resp * sin_phi).sum()
        C_bar_phi = (resp * cos_phi).sum()
        R1 = torch.sqrt(C_bar_phi**2 + S_bar_phi**2)
        mu1 = torch.atan2(S_bar_phi, C_bar_phi)

        S_bar_psi = (resp * sin_psi).sum()
        C_bar_psi = (resp * cos_psi).sum()
        R2 = torch.sqrt(C_bar_psi**2 + S_bar_psi**2)
        mu2 = torch.atan2(S_bar_psi, C_bar_psi)

        # Compute kappas
        kappa1 = (1.28 - 0.53 * R1**2) * torch.tan(0.5 * torch.pi * R1)
        kappa2 = (1.28 - 0.53 * R2**2) * torch.tan(0.5 * torch.pi * R2)

        # Compute lambda
        lambda_ = (kappa1 * kappa2 * (resp * torch.sin(phi - mu1) * torch.sin(psi - mu2)).sum()) / (
            torch.special.i1(kappa1) / torch.special.i0(kappa1) * 
            torch.special.i1(kappa2) / torch.special.i0(kappa2)
        )

        return mu1, mu2, kappa1, kappa2, lambda_

    def _m_step_vect(self, data, responsibilities):
        weights = responsibilities.sum(dim=0) / responsibilities.size(0)

        # Precompute sin and cos values
        sin_phi, cos_phi = torch.sin(data[:, 0]), torch.cos(data[:, 0])
        sin_psi, cos_psi = torch.sin(data[:, 1]), torch.cos(data[:, 1])
        # Expand inputs to match responsibility shape (n_components, n_samples)
        phi = data[:, 0].unsqueeze(0).expand(self.n_components, -1)
        psi = data[:, 1].unsqueeze(0).expand(self.n_components, -1)
        sin_phi = sin_phi.unsqueeze(0).expand(self.n_components, -1)
        cos_phi = cos_phi.unsqueeze(0).expand(self.n_components, -1)
        sin_psi = sin_psi.unsqueeze(0).expand(self.n_components, -1)
        cos_psi = cos_psi.unsqueeze(0).expand(self.n_components, -1)
        # Vectorized computation
        means_kappas = vmap(self._compute_cluster_params)(responsibilities.T, phi, psi, sin_phi, cos_phi, sin_psi, cos_psi)
        
        means = torch.stack([means_kappas[0], means_kappas[1]], dim=1)
        kappas = torch.stack([means_kappas[2], means_kappas[3], means_kappas[4]], dim=1)

        return weights, means, kappas

    def _m_step(self, data, sin_data, cos_data, responsibilities):
        # determine new cluster weights based on expectation step
        weights = responsibilities.sum(dim=0) / responsibilities.size(0)
        # declare empty parameter tensors
        means = torch.empty((self.n_components,2),device=self.device, dtype=self.dtype)
        kappas = torch.empty((self.n_components,3),device=self.device, dtype=self.dtype)
        # determine new values of parameters
        for k in range(self.n_components):
            # Normalize frame weights for each cluster
            resp = responsibilities[:, k]
            resp /= resp.sum()
            # determine ML estimate of mu1
            S_bar = (resp * sin_data[:,0]).sum()
            C_bar = (resp * cos_data[:,0]).sum()
            R1 = torch.sqrt(C_bar**2+S_bar**2) 
            means[k,0] = torch.atan2(S_bar, C_bar)
            # determine ML estimate of mu2
            S_bar = (resp * sin_data[:,1]).sum()
            C_bar = (resp * cos_data[:,1]).sum()
            R2 = torch.sqrt(C_bar**2+S_bar**2) 
            means[k,1] = torch.atan2(S_bar, C_bar)
            # determine ML estimate of kappas currently assuming lambda is small (i.e. truncate sum at m=0)
            # use f4 approximation (min error in Table 1) from Dobson 1978 to determine estimate of kappa1 and kappa2 - this assumes lambda is zero
            kappas[k,0] = (1.28-0.53*R1**2)*torch.tan(0.5*torch.pi*R1)
            kappas[k,1] = (1.28-0.53*R2**2)*torch.tan(0.5*torch.pi*R2)
            # ML estimate of lambda
            # Compute Bessel ratios
            bessel_ratio_k1 = torch.special.i1(kappas[k,0]) / torch.special.i0(kappas[k,0])
            bessel_ratio_k2 = torch.special.i1(kappas[k,1]) / torch.special.i0(kappas[k,1])
            # Weighted estimate for lambda
            kappas[k,2] = (kappas[k,0] * kappas[k,1] * (resp * torch.sin(data[:, 0] - means[k,0]) * torch.sin(data[:, 1] - means[k,1])).sum()) / (bessel_ratio_k1 * bessel_ratio_k2)

        return weights, means, kappas

    def fit(self, data):
        """ Fit the model parameters using data """
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        # Precompute sin and cos values
        sin_data = torch.sin(data)
        cos_data = torch.cos(data)
        # initialize Model parameters
        self.weights_ = torch.full((self.n_components,), 1.0 / self.n_components, device=self.device, dtype=self.dtype)
        self.means_ = torch.rand((self.n_components,2), device=self.device, dtype=self.dtype) * 2 * torch.pi - torch.pi
        self.kappas_ = torch.column_stack( (torch.ones((self.n_components,2), device=self.device, dtype=self.dtype),torch.zeros(self.n_components, device=self.device, dtype=self.dtype)))

        # perform EM:
        old_ll = 0.0
        with torch.no_grad():  # Disable gradients
            for _ in range(self.max_iter):
                responsibilities, ll = self._e_step(data)
                self.weights_, self.means_, self.kappas_ = self._m_step(data, sin_data, cos_data, responsibilities)
                #self.weights_, self.means_, self.kappas_ = self._m_step_vect(data, responsibilities)
            
                if self.verbose:
                    print(_, ll.cpu().numpy(), self.weights_.cpu().numpy())
            
                if torch.abs(ll - old_ll) < self.tol:
                    break
                old_ll = ll
        self.ll = ll


    def predict(self, data):
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._bvm_sine_ln_pdf(data[:,0], data[:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        return ln_likelihoods_per_cluster.argmax(dim=1)

    def ln_pdf(self, data):
        """ Compute ln pdf of the mixture for points """
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._bvm_sine_ln_pdf(data[:,0], data[:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        return torch.logsumexp(ln_likelihoods_per_cluster,1)

    def pdf(self, data):
        """ Compute pdf of the mixture for points """
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        likelihoods_per_cluster = torch.stack([
            self.weights_[k] * self._bvm_sine_pdf(data[:,0], data[:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        return torch.sum(likelihoods_per_cluster,1).cpu().numpy()

    def icl(self, data):
        """Integrated completed likelihood per frame (McNicholas eq 4 and need to read citations)"""
        n_frames = data.shape[0]
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ln_likelihoods_per_cluster = torch.stack([
            torch.log(self.weights_[k]) + self._bvm_sine_ln_pdf(data[:,0], data[:,1], self.means_[k], self.kappas_[k]) 
            for k in range(self.n_components)
        ], dim=1)
        ll = torch.mean(torch.logsumexp(ln_likelihoods_per_cluster,1)).cpu().numpy()
        temp = torch.mean(torch.amax(ln_likelihoods_per_cluster,axis=1)).cpu().numpy()
        return self.n_components*6*np.log(n_frames)/n_frames - 2*ll - 2*temp

    def bic(self, data):
        """Bayesian Information Criterion per frame"""
        n_frames = data.shape[0]
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return self.n_components*6*np.log(n_frames)/n_frames - 2*ll 

    def aic(self, data):
        """Akaike Information Criterion per frame"""
        n_frames = data.shape[0]
        data = torch.tensor(data,device=self.device, dtype=self.dtype)
        ll = self._e_step(data)[1].cpu().numpy()
        return self.n_components*12/n_frames - 2*ll 
    
    def plot_clusters(self, data):
        fontsize=12
        clusters = self.predict(data).cpu()
        plt.figure(figsize=(6, 6))
        plt.scatter(data[:, 0], data[:, 1], c=clusters.numpy(), cmap='viridis', s=50)
        plt.xlabel(r'$\phi$ (radians)',fontsize=fontsize)
        plt.ylabel(r'$\psi$ (radians)',fontsize=fontsize)
        title = "BVM Mixture Model with " + str(self.n_components) + " components"
        plt.title(title, fontsize=fontsize)
        plt.grid(True)
        plt.tick_params(labelsize=fontsize)
        plt.xlim(-np.pi,np.pi)
        plt.ylim(-np.pi,np.pi)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
