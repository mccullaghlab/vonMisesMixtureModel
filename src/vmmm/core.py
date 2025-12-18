import numpy as np
from scipy.special import i0, i1, logsumexp
from scipy.stats import vonmises


class VonMisesMixture:
    """
    EM algorithm for a 1D von Mises mixture model.
    Angles are assumed to be in radians.
    """

    def __init__(self, n_components, max_iter=200, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)

        # Parameters to be learned
        self.weights_ = None   # shape (K,)
        self.mu_ = None        # shape (K,)
        self.kappa_ = None     # shape (K,)

        # Diagnostics
        self.n_iter_ = 0
        self.converged_ = False
        self.lower_bound_ = -np.inf

    # ---------- helpers ----------

    @staticmethod
    def _wrap_angles(theta):
        """Wrap angles to [-pi, pi)."""
        return np.angle(np.exp(1j * theta))

    @staticmethod
    def _A1(kappa):
        """A1(kappa) = I1(kappa)/I0(kappa)."""
        return i1(kappa) / i0(kappa)

    @staticmethod
    def _A1_inv(R):
        """
        Approximate inverse of A1(kappa) using standard Best & Fisher-style
        approximations. R must be in [0, 1).
        """
        R = np.asarray(R)
        kappa = np.zeros_like(R)

        # Avoid divide-by-zero / invalid R
        R_clipped = np.clip(R, 1e-10, 0.999999)

        # Piecewise approximation
        mask1 = R_clipped < 0.53
        mask2 = (R_clipped >= 0.53) & (R_clipped < 0.85)
        mask3 = R_clipped >= 0.85

        # Region 1
        if np.any(mask1):
            R1 = R_clipped[mask1]
            kappa[mask1] = 2 * R1 + R1**3 + (5 * R1**5) / 6.0

        # Region 2
        if np.any(mask2):
            R2 = R_clipped[mask2]
            kappa[mask2] = -0.4 + 1.39 * R2 + 0.43 / (1 - R2)

        # Region 3
        if np.any(mask3):
            R3 = R_clipped[mask3]
            kappa[mask3] = 1.0 / (R3**3 - 4 * R3**2 + 3 * R3)

        # For extremely small R, kappa ~ 0
        kappa[R < 1e-8] = 0.0

        return kappa

    def _log_vonmises_pdf(self, x, mu, kappa):
        """
        Compute log pdf of von Mises for each x, given mu, kappa.
        x: shape (N,)
        mu, kappa: shape (K,)
        Returns: shape (N, K)
        """
        x = x[:, None]  # (N, 1)
        mu = mu[None, :]  # (1, K)
        kappa = kappa[None, :]  # (1, K)

        # log C(kappa) = -log(2*pi*I0(kappa))
        logC = -np.log(2 * np.pi * i0(kappa))
        return kappa * np.cos(x - mu) + logC

    # ---------- core EM ----------

    def _initialize_params(self, x):
        N = x.shape[0]
        K = self.n_components

        # Initialize weights uniformly
        self.weights_ = np.full(K, 1.0 / K)

        # Initialize means by picking random points
        idx = self.random_state.choice(N, K, replace=False)
        self.mu_ = x[idx]

        # Initialize kappas to a moderate concentration
        self.kappa_ = np.full(K, 1.0)

    def _e_step(self, x):
        """Compute responsibilities and log-likelihood."""
        log_pdf = self._log_vonmises_pdf(x, self.mu_, self.kappa_)  # (N, K)
        log_weights = np.log(self.weights_)[None, :]                 # (1, K)

        log_prob = log_pdf + log_weights                            # (N, K)
        log_prob_norm = logsumexp(log_prob, axis=1)                 # (N,)
        log_resp = log_prob - log_prob_norm[:, None]                # (N, K)
        resp = np.exp(log_resp)

        # Average log-likelihood
        ll = log_prob_norm.mean()
        return resp, ll

    def _m_step(self, x, resp):
        """Update weights, mu, kappa from responsibilities."""
        N, K = resp.shape

        # Effective counts
        Nk = resp.sum(axis=0)  # shape (K,)

        # Update weights
        self.weights_ = Nk / N

        # Update mu and kappa for each component
        # Weighted sums of cos and sin
        Ck = np.sum(resp * np.cos(x[:, None]), axis=0)
        Sk = np.sum(resp * np.sin(x[:, None]), axis=0)

        self.mu_ = np.arctan2(Sk, Ck)

        Rk = np.sqrt(Ck**2 + Sk**2) / Nk  # mean resultant length per comp
        self.kappa_ = self._A1_inv(Rk)

    def fit(self, x):
        """
        Fit the mixture model to data x (1D angles in radians).
        """
        x = np.asarray(x).ravel()
        x = self._wrap_angles(x)

        self._initialize_params(x)

        lower_bound_old = -np.inf

        for n_iter in range(1, self.max_iter + 1):
            resp, ll = self._e_step(x)
            self._m_step(x, resp)

            self.lower_bound_ = ll
            change = ll - lower_bound_old

            if np.abs(change) < self.tol:
                self.converged_ = True
                break

            lower_bound_old = ll

        self.n_iter_ = n_iter
        return self

    # ---------- utilities ----------

    def predict_proba(self, x):
        """
        Posterior responsibilities P(component | x).
        """
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        x = np.asarray(x).ravel()
        x = self._wrap_angles(x)

        log_pdf = self._log_vonmises_pdf(x, self.mu_, self.kappa_)  # (N, K)
        log_weights = np.log(self.weights_)[None, :]
        log_prob = log_pdf + log_weights
        log_prob_norm = logsumexp(log_prob, axis=1)
        log_resp = log_prob - log_prob_norm[:, None]
        return np.exp(log_resp)

    def predict(self, x):
        """
        Hard assignments: argmax over responsibilities.
        """
        return np.argmax(self.predict_proba(x), axis=1)

    def score_samples(self, x):
        """
        Log-likelihood per sample: log p(x_i).
        """
        x = np.asarray(x).ravel()
        x = self._wrap_angles(x)

        log_pdf = self._log_vonmises_pdf(x, self.mu_, self.kappa_)
        log_weights = np.log(self.weights_)[None, :]
        log_prob = log_pdf + log_weights
        return logsumexp(log_prob, axis=1)

    def score(self, x):
        """
        Return the average log-likelihood of the data under the model.
        (Equivalent to sklearn.mixture.GaussianMixture.score)

        Parameters
        ----------
        x : array-like, shape (N,)
            Angle data in radians.

        Returns
        -------
        float
            Mean log-likelihood per sample.
        """
        logp = self.score_samples(x)   # shape (N,)
        return np.mean(logp)


    def sample(self, n_samples=1):
        """
        Draw samples from the fitted mixture.
        Returns: samples of shape (n_samples,), and component labels.
        """
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")

        K = self.n_components
        # Choose components according to weights
        comp = self.random_state.choice(K, size=n_samples, p=self.weights_)

        samples = np.empty(n_samples)
        for k in range(K):
            mask = comp == k
            n_k = np.sum(mask)
            if n_k > 0:
                samples[mask] = vonmises.rvs(self.kappa_[k],
                                             loc=self.mu_[k],
                                             size=n_k,
                                             random_state=self.random_state)
        return samples

