import numpy as np
from . import bvvmmm
import matplotlib.pyplot as plt
import torch
import sys
import warnings

def pick_elbow_component(ll, component_range, thresh = 0.01):

    #compute gradient of "normalized" log likelihood
    grad = np.gradient(ll/(np.amax(ll) - np.amin(ll)))
    # Elbow is the point right before fraction of information gained goes below the threshold
    if np.any(grad < thresh):
        elbow_index = np.amin(np.argwhere(grad < thresh)) - 1
    # else set to last index
    else:
        elbow_index = -1
    return component_range[elbow_index]

def assert_radians(data, lower_bound=-np.pi, upper_bound=np.pi):
    # Flatten the data for a global check
    data_flat = data.flatten()
    # Check if the majority of the data falls outside the expected radian range
    if np.any(data_flat < lower_bound) or np.any(data_flat > upper_bound):
        warnings.warn("Data values appear to be outside the typical radian range "
                      f"[{lower_bound}, {upper_bound}]. Ensure that the data is provided in radians.")
        sys.exit(1)

class MultiIndSineBVvMMM:
    """
    Sine Bivariate von Mises Mixture Model Expectation-Maximization (EM) Algorithm

    This class implements an Expectation-Maximization (EM) algorithm for
    fitting a mixture model of Sine Bivariate von Mises (BVM) distributions
    to circular data, such as protein backbone dihedral angles (phi, psi).

    Parameters
    ----------

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
    >>> model = MultiIndSineBVvMMM(n_components=3, max_iter=200, verbose=True, tol=1e-5, seed=1234)
    >>> model.fit(data)
    >>> model.plot_clusters(data)
    """

    def __init__(self, max_iter=100, tol=1e-4, device=None, dtype=torch.float64, seed=None, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
        self.verbose = verbose
        self.scan_flag_ = False
        self.fit_flag_ = False
        self.weights_ = None
        self.kappas_ = None
        self.means_ = None

    def component_scan(self, data, components_scan=np.arange(1,8,1), n_attempts=15, train_frac=0.95):

        self.components_scan = components_scan
        self.n_components_scan = components_scan.size
        self.n_residues = data.shape[1]
        self.ll_scan = np.empty((self.n_components_scan,self.n_residues))
        self.cv_ll_scan = np.empty((self.n_components_scan,self.n_residues))
        self.icl_scan = np.empty((self.n_components_scan,self.n_residues))
        self.components = np.empty(self.n_residues,dtype=np.int64)
        # plot parameters
        fontsize=12
        for residue in range(self.n_residues):
            self.ll_scan[:,residue], _, _, self.icl_scan[:,residue], self.cv_ll_scan[:,residue], best_models = bvvmmm.component_scan(data[:,residue,:], self.components_scan, n_attempts=n_attempts, tol=self.tol, device=self.device, dtype=self.dtype, train_frac=train_frac, verbose=False)
            self.components[residue] = pick_elbow_component(self.ll_scan[:,residue], self.components_scan)
            # plot!!!
            fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 rows, 2 columns
            # LL phi/psi 1
            axes[0].plot(self.components_scan, self.ll_scan[:,residue], '-o', lw=3)
            axes[0].plot(self.components_scan, self.cv_ll_scan[:,residue], '--', lw=2)
            title = "Residue " + str(residue+1) + " LL vs n_components"
            axes[0].set_title(title, fontsize=fontsize)
            axes[0].set_xlabel('Number of Components',fontsize=fontsize)
            axes[0].set_ylabel('LL',fontsize=fontsize)
            axes[0].grid(True)
            axes[0].tick_params(axis='both', labelsize=fontsize)
            # attempt to circle the elbow
            axes[0].scatter(self.components[residue],self.ll_scan[self.components_scan == self.components[residue],residue], s=500, marker='o', facecolors='none', edgecolors='r', linewidths=3)
            # circle the elbow
            #circle = patches.Circle((self.components[residue], self.ll_scan[self.components_scan == self.components[residue],residue]), radius, edgecolor='r', facecolor='none')
            # Add the circle to the axes
            #axes[0].add_patch(circle)
            # plot fe
            title = "Residue " + str(residue+1) + " model (color) + sample (contour) FE/kT"
            best_models[np.argwhere(self.components_scan == self.components[residue])[0,0]].plot_model_sample_fe(data[:,residue,:],axes=axes[1],title=title)
            # finish plot
            plt.tight_layout()
            plt.show();
        self.total_components = self.components.sum()
        self.scan_flag_ = True
        print("Suggested components:", self.components)

    def fit(self, data, n_attempts = 15, components=None, verbose=True, plot=False):
        """ fit BVvMMM to each residue with either provided components or components determined by scan """
        # meta data
        n_frames = data.shape[0]
        self.n_residues = data.shape[1]
        if components is not None:
            self.components = components
            self.total_components = components.sum()
        elif self.scan_flag_ == False:
            print("Need to either provide components for each residue or perform a scan")

        # fit each residue to their components
        self.residue_models_ = []
        self.cluster_ids = np.empty((n_frames,self.n_residues),dtype=np.int64)
        for residue in range(self.n_residues):
            if verbose==True:
                print("Fitting Residue ", residue+1)
            model = bvvmmm.fit_with_attempts(data[:,residue,:],self.components[residue],n_attempts, verbose=verbose, device=self.device, dtype=self.dtype)
            self.cluster_ids[:,residue] = model.predict(data[:,residue,:])[0]
            self.residue_models_.append(model)

        # determine "macro" cluster ids and populations
        self.macro_state_ids, self.macro_state_counts = np.unique(self.cluster_ids,axis=0,return_counts=True)
        self.macro_state_weights_ = self.macro_state_counts / n_frames
        self.n_macro_states = self.macro_state_counts.size
        # if requested, compare indpendent expectation to observation of cluster populations
        if plot == True:
            total_possible_states = self.components.prod()
            cum_prod = np.cumprod(self.components[::-1])
            predicted_population = np.ones(total_possible_states)
            observed_population = np.zeros(total_possible_states)
            count = 0
            for residue in range(self.n_residues):
                if residue == 0:
                    residue_range = 1
                else:
                    residue_range = cum_prod[residue-1]
                for count in range(total_possible_states // cum_prod[residue]):
                    for component in range(self.components[-residue-1]):
                        predicted_population[count*cum_prod[residue] + component*residue_range:count*cum_prod[residue] + (component+1)*residue_range] *= self.residue_models_[residue].weights_[component].cpu().numpy()
            # determine observed
            cum_prod = cum_prod[::-1]
            for macro_cluster in range(self.n_macro_states):
                state_id = 0
                for residue in range(self.n_residues-1):
                    state_id += self.macro_state_ids[macro_cluster,residue]*cum_prod[residue+1]
                state_id += self.macro_state_ids[macro_cluster,-1]
                observed_population[state_id] = self.macro_state_weights_[macro_cluster]
            # plot
            plt.figure(figsize=(5,5))
            fontsize=12
            plt.bar(np.arange(total_possible_states),observed_population, label="observed")
            plt.bar(np.arange(total_possible_states),predicted_population, alpha=0.75, label="predicted assuming independent")
            plt.title('Macro State Populations',fontsize=fontsize)
            plt.xlabel('Macro State ID',fontsize=fontsize)
            plt.ylabel('Population',fontsize=fontsize)
            plt.tick_params(axis='both', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            plt.show();
        # update flag
        self.fit_flag_ = True

    def predict(self, data):
        """ predict cluster ids """
        # first check that the object has been fit
        if not self.fit_flag_:
            print("You must fit the object before you can predit")
            sys.exit(1)
        # check data is in radians
        assert_radians(data)
        # pass data to pyTorch
        data = torch.tensor(data, device=self.device, dtype=self.dtype)
        n_samples = data.shape[0]
        # declare ln_pdf list
        ln_pdf_list = []
        for residue in range(self.n_residues):
            # precompute diff_sin_prod
            diff_phi = torch.sin(data[:, residue, 0].unsqueeze(1) - self.residue_models_[residue].means_[:,0].unsqueeze(0))
            diff_psi = torch.sin(data[:, residue, 1].unsqueeze(1) - self.residue_models_[residue].means_[:,1].unsqueeze(0))
            diff_sin_prod = diff_phi * diff_psi
            # Vectorized log-PDF evaluation.
            phi = data[:, residue, 0].unsqueeze(1).expand(n_samples, self.residue_models_[residue].n_components)
            psi = data[:, residue, 1].unsqueeze(1).expand(n_samples, self.residue_models_[residue].n_components)
            # compute ln pdf
            ln_pdf = bvvmmm.batched_bvm_sine_ln_pdf(phi, psi, diff_sin_prod, self.residue_models_[residue].means_, self.residue_models_[residue].kappas_, self.residue_models_[residue].normalization_)
            # add to list
            ln_pdf_list.append(ln_pdf)
        # compute macro state ln_pdf
        macro_ln_pdf = torch.zeros((n_samples,self.n_macro_states), device=self.device, dtype=self.dtype)
        for state in range(self.n_macro_states):
            for residue in range(self.n_residues):
                macro_ln_pdf[:,state] += ln_pdf_list[residue][:,self.macro_state_ids[state,residue]]
        # return cluster ids
        return macro_ln_pdf.argmax(dim=1).cpu().numpy()

    def plot_marginal_fes(self, data):
        """ make plots of marginal fes for each residue """
        # plot parameters
        #fontsize=12
        #fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1 rows, 2 columns
        for residue in range(self.n_residues):
            # plot fe
            title = "Residue " + str(residue+1) + " model (color) + sample (contour) FE/kT"
            self.residue_models_[residue].plot_model_sample_fe(data[:,residue,:], title=title)
